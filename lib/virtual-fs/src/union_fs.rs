//! VFS-inspired union filesystem that owns mount topology.
//!
//! `MountFileSystem` is the primary type. `UnionFileSystem` is a backward-compatible
//! type alias kept for existing callers.

use dashmap::DashMap;

use crate::*;

use std::{
    collections::HashSet,
    ffi::OsString,
    path::{Component, Path},
    sync::{Arc, RwLock},
};

/// A single node in the mount tree.
///
/// A node represents one path component in the mount hierarchy.
/// It may have a filesystem mounted at its exact path (`mount`),
/// and zero or more child mount-points (`children`).
///
/// Using `Arc<MountNode>` as children values lets callers clone the `Arc`
/// cheaply and drop the parent `DashMap` lock before recursing, which avoids
/// holding multiple shard locks at once.
pub struct MountNode {
    mount: RwLock<Option<Arc<dyn FileSystem + Send + Sync>>>,
    children: DashMap<OsString, Arc<MountNode>>,
}

impl std::fmt::Debug for MountNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mounted = self.mount.read().map(|g| g.is_some()).unwrap_or(false);
        let child_keys: Vec<OsString> = self.children.iter().map(|e| e.key().clone()).collect();
        f.debug_struct("MountNode")
            .field("mounted", &mounted)
            .field("children", &child_keys)
            .finish()
    }
}

impl MountNode {
    fn new() -> Self {
        MountNode {
            mount: RwLock::new(None),
            children: DashMap::new(),
        }
    }

    /// Deep-clone this node and all descendants, sharing the mounted `Arc<dyn FileSystem>`
    /// values but creating a new independent tree topology.
    fn deep_clone(self: &Arc<Self>) -> Arc<Self> {
        let fs = self.mount.read().unwrap().clone();
        let new_node = MountNode {
            mount: RwLock::new(fs),
            children: DashMap::new(),
        };
        for entry in self.children.iter() {
            new_node
                .children
                .insert(entry.key().clone(), entry.value().deep_clone());
        }
        Arc::new(new_node)
    }

    /// Mount `fs` at the path described by `components` relative to this node.
    fn mount_at(
        &self,
        components: &[OsString],
        fs: Arc<dyn FileSystem + Send + Sync>,
    ) -> Result<()> {
        if components.is_empty() {
            let mut lock = self.mount.write().unwrap();
            if lock.is_some() {
                return Err(FsError::AlreadyExists);
            }
            *lock = Some(fs);
            Ok(())
        } else {
            // Get-or-create the child node, clone its Arc, then release the
            // DashMap shard lock before recursing.
            let child = {
                let entry_ref = self
                    .children
                    .entry(components[0].clone())
                    .or_insert_with(|| Arc::new(MountNode::new()));
                Arc::clone(&*entry_ref)
            };
            child.mount_at(&components[1..], fs)
        }
    }
}

/// A filesystem that manages a tree of mount points.
///
/// `MountFileSystem` owns mount-point topology: which filesystem is mounted at
/// which path, how path resolution crosses mount boundaries, and how directory
/// listings expose sub-mounts.  Leaf filesystems (`mem_fs`, `host_fs`,
/// `WebcVolumeFileSystem`, …) only need to implement operations for their own
/// trees.
///
/// Path resolution always delegates to the *deepest* mounted node, so nested
/// mounts work even when an intermediate leaf filesystem does not implement
/// `mount()`.
///
/// Concurrent reads are lock-free at the tree level: each node holds an
/// `RwLock` only for its own `mount` slot, and `DashMap` shards are released
/// before any recursion.
#[derive(Clone)]
pub struct MountFileSystem {
    root: Arc<MountNode>,
}

impl std::fmt::Debug for MountFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MountFileSystem")
            .field("root", &self.root)
            .finish()
    }
}

impl Default for MountFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward-compatible alias.  New code should prefer `MountFileSystem`.
pub type UnionFileSystem = MountFileSystem;

/// Defines how to handle conflicts when merging two [`MountFileSystem`]s.
#[derive(Clone, Copy, Debug)]
pub enum UnionMergeMode {
    /// Replace existing mount slots with the incoming ones.
    Replace,
    /// Keep existing mount slots; skip the incoming ones silently.
    Skip,
    /// Return [`FsError::AlreadyExists`] if a conflict is found.
    Fail,
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn normalize_path_components(path: &Path) -> Vec<OsString> {
    path.components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_os_string()),
            _ => None,
        })
        .collect()
}

fn directory_metadata() -> Metadata {
    Metadata {
        ft: FileType::new_dir(),
        accessed: 0,
        created: 0,
        modified: 0,
        len: 0,
    }
}

fn rebase_entries(entries: &mut ReadDir, prefix: &Path) {
    for entry in &mut entries.data {
        let suffix: PathBuf = entry.path.components().skip(1).collect();
        entry.path = prefix.join(suffix);
    }
}

// ── MountFileSystem impl ──────────────────────────────────────────────────────

impl MountFileSystem {
    pub fn new() -> Self {
        MountFileSystem {
            root: Arc::new(MountNode::new()),
        }
    }

    /// Clear all mounts, replacing the root with a fresh empty node.
    pub fn clear(&mut self) {
        self.root = Arc::new(MountNode::new());
    }

    /// Create an independent copy of this filesystem's mount topology.
    ///
    /// The returned filesystem shares the same leaf `Arc<dyn FileSystem>`
    /// values (the filesystems themselves are shared) but has a completely
    /// independent mount tree, so adding or removing mounts in one copy does
    /// not affect the other.
    ///
    /// This differs from [`Clone`], which produces a shallow copy that shares
    /// the same root node.
    pub fn duplicate(&self) -> Self {
        MountFileSystem {
            root: self.root.deep_clone(),
        }
    }

    /// Merge `other` into `self` according to `mode`.
    pub fn merge(&self, other: &MountFileSystem, mode: UnionMergeMode) -> Result<()> {
        Self::merge_nodes(&self.root, &other.root, mode)
    }

    fn merge_nodes(dest: &MountNode, src: &MountNode, mode: UnionMergeMode) -> Result<()> {
        // Merge the mount slot.
        {
            let src_guard = src.mount.read().unwrap();
            if let Some(src_fs) = src_guard.as_ref() {
                let mut dest_guard = dest.mount.write().unwrap();
                match dest_guard.as_ref() {
                    Some(_) => match mode {
                        UnionMergeMode::Replace => {
                            *dest_guard = Some(Arc::clone(src_fs));
                        }
                        UnionMergeMode::Skip => {
                            tracing::debug!(
                                "skipping existing mount point while merging two filesystems"
                            );
                        }
                        UnionMergeMode::Fail => return Err(FsError::AlreadyExists),
                    },
                    None => {
                        *dest_guard = Some(Arc::clone(src_fs));
                    }
                }
            }
        }

        // Collect src children before recursing to avoid holding DashMap
        // references across recursive calls.
        let src_children: Vec<(OsString, Arc<MountNode>)> = src
            .children
            .iter()
            .map(|e| (e.key().clone(), Arc::clone(e.value())))
            .collect();

        for (name, src_child) in src_children {
            let dest_child = {
                let entry_ref = dest
                    .children
                    .entry(name)
                    .or_insert_with(|| Arc::new(MountNode::new()));
                Arc::clone(&*entry_ref)
            };
            Self::merge_nodes(&dest_child, &src_child, mode)?;
        }

        Ok(())
    }

    /// Walk the mount tree and find the deepest mounted node whose path is a
    /// prefix of `path`.  Returns the delegated filesystem and the suffix
    /// path to use within it.
    fn resolve_deepest_mount(
        &self,
        path: &Path,
    ) -> Option<(Arc<dyn FileSystem + Send + Sync>, PathBuf)> {
        let components = normalize_path_components(path);
        let mut best_fs: Option<Arc<dyn FileSystem + Send + Sync>> = None;
        let mut best_consumed: usize = 0;

        // Check root node.
        {
            let guard = self.root.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                best_fs = Some(Arc::clone(fs));
                best_consumed = 0;
            }
        }

        let mut node = Arc::clone(&self.root);
        for (i, component) in components.iter().enumerate() {
            // Clone the Arc and release the DashMap lock before the next
            // iteration so we never hold two shard locks simultaneously.
            let child = node
                .children
                .get(component.as_os_str())
                .map(|r| Arc::clone(r.value()));
            match child {
                None => break,
                Some(child_node) => {
                    {
                        let guard = child_node.mount.read().unwrap();
                        if let Some(fs) = guard.as_ref() {
                            best_fs = Some(Arc::clone(fs));
                            best_consumed = i + 1;
                        }
                    }
                    node = child_node;
                }
            }
        }

        best_fs.map(|fs| {
            let remaining: PathBuf = components[best_consumed..].iter().collect();
            (fs, Path::new("/").join(remaining))
        })
    }

    /// Find the exact `MountNode` for `path`, if one exists in the tree.
    fn find_node(&self, path: &Path) -> Option<Arc<MountNode>> {
        let components = normalize_path_components(path);
        let mut node = Arc::clone(&self.root);

        for component in &components {
            let child = node
                .children
                .get(component.as_os_str())
                .map(|r| Arc::clone(r.value()));
            match child {
                None => return None,
                Some(child_node) => node = child_node,
            }
        }

        Some(node)
    }
}

// ── FileSystem trait ──────────────────────────────────────────────────────────

impl FileSystem for MountFileSystem {
    fn readlink(&self, path: &Path) -> Result<PathBuf> {
        match self.resolve_deepest_mount(path) {
            Some((fs, delegated)) => fs.readlink(&delegated),
            None => Err(FsError::EntryNotFound),
        }
    }

    fn read_dir(&self, path: &Path) -> Result<ReadDir> {
        let components = normalize_path_components(path);
        let prefix = PathBuf::from("/").join(components.iter().collect::<PathBuf>());

        match self.find_node(path) {
            Some(node) => {
                // Exact node found: merge the mounted fs's root entries with
                // the names of any child sub-mounts.  Child mount names shadow
                // same-named entries from the base filesystem.
                let child_names: HashSet<OsString> =
                    node.children.iter().map(|e| e.key().clone()).collect();
                let mut entries = Vec::new();

                {
                    let guard = node.mount.read().unwrap();
                    if let Some(fs) = guard.as_ref() {
                        let mut base = fs.read_dir(Path::new("/"))?;
                        rebase_entries(&mut base, &prefix);
                        entries.extend(base.data.into_iter().filter(|entry| {
                            entry
                                .path
                                .file_name()
                                .map(|n| !child_names.contains(n))
                                .unwrap_or(true)
                        }));
                    }
                }

                entries.extend(child_names.into_iter().map(|name| DirEntry {
                    path: prefix.join(PathBuf::from(&name)),
                    metadata: Ok(directory_metadata()),
                }));

                Ok(ReadDir::new(entries))
            }
            None => {
                // Path is not a mount-tree node; delegate to the deepest mount.
                match self.resolve_deepest_mount(path) {
                    Some((fs, delegated)) => {
                        let mut entries = fs.read_dir(&delegated)?;
                        rebase_entries(&mut entries, &prefix);
                        Ok(entries)
                    }
                    None => Err(FsError::EntryNotFound),
                }
            }
        }
    }

    fn create_dir(&self, path: &Path) -> Result<()> {
        let components = normalize_path_components(path);

        if components.is_empty() {
            // Creating the virtual root is always a no-op.
            return Ok(());
        }

        if let Some(node) = self.find_node(path) {
            let guard = node.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                match fs.create_dir(Path::new("/")) {
                    Err(FsError::AlreadyExists) => Ok(()),
                    other => other,
                }
            } else {
                // Branch-only node: the directory implicitly exists.
                Ok(())
            }
        } else {
            match self.resolve_deepest_mount(path) {
                Some((fs, delegated)) => match fs.create_dir(&delegated) {
                    Err(FsError::AlreadyExists) => Ok(()),
                    other => other,
                },
                None => Err(FsError::EntryNotFound),
            }
        }
    }

    fn remove_dir(&self, path: &Path) -> Result<()> {
        let components = normalize_path_components(path);

        if components.is_empty() {
            return Err(FsError::PermissionDenied);
        }

        if let Some(node) = self.find_node(path) {
            if !node.children.is_empty() {
                // Refuse to remove a node that still has child mounts.
                return Err(FsError::PermissionDenied);
            }
            let guard = node.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                fs.remove_dir(Path::new("/"))
            } else {
                Err(FsError::EntryNotFound)
            }
        } else {
            match self.resolve_deepest_mount(path) {
                Some((fs, delegated)) => fs.remove_dir(&delegated),
                None => Err(FsError::EntryNotFound),
            }
        }
    }

    fn rename<'a>(&'a self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let from_components = normalize_path_components(from);

            if from_components.is_empty() {
                return Err(FsError::PermissionDenied);
            }

            // Refuse to rename a mount-tree node that is branch-only or has
            // child mounts.
            if let Some(node) = self.find_node(from) {
                let has_children = !node.children.is_empty();
                let guard = node.mount.read().unwrap();
                if guard.is_none() || has_children {
                    return Err(FsError::PermissionDenied);
                }
            }

            match (
                self.resolve_deepest_mount(from),
                self.resolve_deepest_mount(to),
            ) {
                // Two paths are in the same mounted filesystem when they
                // resolve to the same Arc pointer.  Each `mount()` call
                // stores exactly one Arc per mount slot, so paths under the
                // same mount always produce the same pointer.  Cross-mount
                // renames are rejected with InvalidInput.
                (Some((from_fs, from_path)), Some((to_fs, to_path)))
                    if Arc::ptr_eq(&from_fs, &to_fs) =>
                {
                    from_fs.rename(&from_path, &to_path).await
                }
                (Some(_), Some(_)) => Err(FsError::InvalidInput),
                _ => Err(FsError::EntryNotFound),
            }
        })
    }

    fn metadata(&self, path: &Path) -> Result<Metadata> {
        let components = normalize_path_components(path);

        if components.is_empty() {
            let guard = self.root.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                fs.metadata(Path::new("/"))
            } else {
                Ok(directory_metadata())
            }
        } else if let Some(node) = self.find_node(path) {
            let guard = node.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                fs.metadata(Path::new("/"))
            } else {
                // Branch-only node: synthetic directory metadata.
                Ok(directory_metadata())
            }
        } else {
            match self.resolve_deepest_mount(path) {
                Some((fs, delegated)) => fs.metadata(&delegated),
                None => Err(FsError::EntryNotFound),
            }
        }
    }

    fn symlink_metadata(&self, path: &Path) -> Result<Metadata> {
        let components = normalize_path_components(path);

        if components.is_empty() {
            let guard = self.root.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                fs.symlink_metadata(Path::new("/"))
            } else {
                Ok(directory_metadata())
            }
        } else if let Some(node) = self.find_node(path) {
            let guard = node.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                fs.symlink_metadata(Path::new("/"))
            } else {
                Ok(directory_metadata())
            }
        } else {
            match self.resolve_deepest_mount(path) {
                Some((fs, delegated)) => fs.symlink_metadata(&delegated),
                None => Err(FsError::EntryNotFound),
            }
        }
    }

    fn remove_file(&self, path: &Path) -> Result<()> {
        let components = normalize_path_components(path);

        if components.is_empty() {
            return Err(FsError::NotAFile);
        }

        if let Some(node) = self.find_node(path) {
            if !node.children.is_empty() {
                return Err(FsError::PermissionDenied);
            }
            let guard = node.mount.read().unwrap();
            if let Some(fs) = guard.as_ref() {
                fs.remove_file(Path::new("/"))
            } else {
                Err(FsError::EntryNotFound)
            }
        } else {
            match self.resolve_deepest_mount(path) {
                Some((fs, delegated)) => fs.remove_file(&delegated),
                None => Err(FsError::EntryNotFound),
            }
        }
    }

    fn new_open_options(&self) -> OpenOptions<'_> {
        OpenOptions::new(self)
    }

    fn mount(
        &self,
        // The `name` parameter is part of the `FileSystem` trait signature and
        // retained for backward compatibility.  `MountFileSystem` does not
        // store names; mount topology is keyed by path alone.
        _name: String,
        path: &Path,
        fs: Box<dyn FileSystem + Send + Sync>,
    ) -> Result<()> {
        let components = normalize_path_components(path);
        if components.is_empty() {
            // Mounting at root.
            let mut guard = self.root.mount.write().unwrap();
            if guard.is_some() {
                return Err(FsError::AlreadyExists);
            }
            *guard = Some(Arc::from(fs));
            Ok(())
        } else {
            self.root.mount_at(&components, Arc::from(fs))
        }
    }
}

// ── FileOpener ────────────────────────────────────────────────────────────────

impl FileOpener for MountFileSystem {
    fn open(
        &self,
        path: &Path,
        conf: &OpenOptionsConfig,
    ) -> Result<Box<dyn VirtualFile + Send + Sync>> {
        // A branch-only mount-tree node (no mounted fs) is a virtual directory,
        // not a file.
        if let Some(node) = self.find_node(path) {
            let guard = node.mount.read().unwrap();
            if guard.is_none() {
                return Err(FsError::NotAFile);
            }
        }

        match self.resolve_deepest_mount(path) {
            Some((fs, delegated)) => fs.new_open_options().options(conf.clone()).open(delegated),
            None => Err(FsError::EntryNotFound),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        path::{Path, PathBuf},
    };

    use tokio::io::AsyncWriteExt;

    use crate::{
        FileOpener, FileSystem as FileSystemTrait, FsError, OpenOptionsConfig, UnionFileSystem,
        UnionMergeMode, mem_fs,
    };

    #[derive(Debug, Clone, Default)]
    struct MountlessFileSystem {
        inner: mem_fs::FileSystem,
    }

    impl FileSystemTrait for MountlessFileSystem {
        fn readlink(&self, path: &Path) -> crate::Result<PathBuf> {
            self.inner.readlink(path)
        }

        fn read_dir(&self, path: &Path) -> crate::Result<crate::ReadDir> {
            self.inner.read_dir(path)
        }

        fn create_dir(&self, path: &Path) -> crate::Result<()> {
            self.inner.create_dir(path)
        }

        fn remove_dir(&self, path: &Path) -> crate::Result<()> {
            self.inner.remove_dir(path)
        }

        fn rename<'a>(
            &'a self,
            from: &'a Path,
            to: &'a Path,
        ) -> futures::future::BoxFuture<'a, crate::Result<()>> {
            Box::pin(async move { self.inner.rename(from, to).await })
        }

        fn metadata(&self, path: &Path) -> crate::Result<crate::Metadata> {
            self.inner.metadata(path)
        }

        fn symlink_metadata(&self, path: &Path) -> crate::Result<crate::Metadata> {
            self.inner.symlink_metadata(path)
        }

        fn remove_file(&self, path: &Path) -> crate::Result<()> {
            self.inner.remove_file(path)
        }

        fn new_open_options(&self) -> crate::OpenOptions<'_> {
            self.inner.new_open_options()
        }

        fn mount(
            &self,
            _name: String,
            _path: &Path,
            _fs: Box<dyn FileSystemTrait + Send + Sync>,
        ) -> crate::Result<()> {
            Err(FsError::Unsupported)
        }
    }

    impl FileOpener for MountlessFileSystem {
        fn open(
            &self,
            path: &Path,
            conf: &OpenOptionsConfig,
        ) -> crate::Result<Box<dyn crate::VirtualFile + Send + Sync>> {
            self.inner
                .new_open_options()
                .options(conf.clone())
                .open(path)
        }
    }

    fn gen_filesystem() -> UnionFileSystem {
        let union = UnionFileSystem::new();
        let a = mem_fs::FileSystem::default();
        let b = mem_fs::FileSystem::default();
        let c = mem_fs::FileSystem::default();
        let d = mem_fs::FileSystem::default();
        let e = mem_fs::FileSystem::default();
        let f = mem_fs::FileSystem::default();
        let g = mem_fs::FileSystem::default();
        let h = mem_fs::FileSystem::default();

        union
            .mount(
                "mem_fs_1".to_string(),
                PathBuf::from("/test_new_filesystem").as_path(),
                Box::new(a),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_2".to_string(),
                PathBuf::from("/test_create_dir").as_path(),
                Box::new(b),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_3".to_string(),
                PathBuf::from("/test_remove_dir").as_path(),
                Box::new(c),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_4".to_string(),
                PathBuf::from("/test_rename").as_path(),
                Box::new(d),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_5".to_string(),
                PathBuf::from("/test_metadata").as_path(),
                Box::new(e),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_6".to_string(),
                PathBuf::from("/test_remove_file").as_path(),
                Box::new(f),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_6".to_string(),
                PathBuf::from("/test_readdir").as_path(),
                Box::new(g),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_6".to_string(),
                PathBuf::from("/test_canonicalize").as_path(),
                Box::new(h),
            )
            .unwrap();

        union
    }

    fn gen_nested_filesystem() -> UnionFileSystem {
        let union = UnionFileSystem::new();
        let a = mem_fs::FileSystem::default();
        a.open(
            &PathBuf::from("/data-a.txt"),
            &OpenOptionsConfig {
                read: true,
                write: true,
                create_new: false,
                create: true,
                append: false,
                truncate: false,
            },
        )
        .unwrap();
        let b = mem_fs::FileSystem::default();
        b.open(
            &PathBuf::from("/data-b.txt"),
            &OpenOptionsConfig {
                read: true,
                write: true,
                create_new: false,
                create: true,
                append: false,
                truncate: false,
            },
        )
        .unwrap();

        union
            .mount(
                "mem_fs_1".to_string(),
                PathBuf::from("/app/a").as_path(),
                Box::new(a),
            )
            .unwrap();
        union
            .mount(
                "mem_fs_2".to_string(),
                PathBuf::from("/app/b").as_path(),
                Box::new(b),
            )
            .unwrap();

        union
    }

    #[tokio::test]
    async fn test_nested_read_dir() {
        let fs = gen_nested_filesystem();

        let root_contents: Vec<PathBuf> = fs
            .read_dir(&PathBuf::from("/"))
            .unwrap()
            .map(|e| e.unwrap().path.clone())
            .collect();
        assert_eq!(root_contents, vec![PathBuf::from("/app")]);

        let app_contents: HashSet<PathBuf> = fs
            .read_dir(&PathBuf::from("/app"))
            .unwrap()
            .map(|e| e.unwrap().path)
            .collect();
        assert_eq!(
            app_contents,
            HashSet::from_iter([PathBuf::from("/app/a"), PathBuf::from("/app/b")].into_iter())
        );

        let a_contents: Vec<PathBuf> = fs
            .read_dir(&PathBuf::from("/app/a"))
            .unwrap()
            .map(|e| e.unwrap().path.clone())
            .collect();
        assert_eq!(a_contents, vec![PathBuf::from("/app/a/data-a.txt")]);

        let b_contents: Vec<PathBuf> = fs
            .read_dir(&PathBuf::from("/app/b"))
            .unwrap()
            .map(|e| e.unwrap().path)
            .collect();
        assert_eq!(b_contents, vec![PathBuf::from("/app/b/data-b.txt")]);
    }

    #[tokio::test]
    async fn test_nested_metadata() {
        let fs = gen_nested_filesystem();

        assert!(fs.metadata(&PathBuf::from("/")).is_ok());
        assert!(fs.metadata(&PathBuf::from("/app")).is_ok());
        assert!(fs.metadata(&PathBuf::from("/app/a")).is_ok());
        assert!(fs.metadata(&PathBuf::from("/app/b")).is_ok());
        assert!(fs.metadata(&PathBuf::from("/app/a/data-a.txt")).is_ok());
        assert!(fs.metadata(&PathBuf::from("/app/b/data-b.txt")).is_ok());
    }

    #[tokio::test]
    async fn test_nested_symlink_metadata() {
        let fs = gen_nested_filesystem();

        assert!(fs.symlink_metadata(&PathBuf::from("/")).is_ok());
        assert!(fs.symlink_metadata(&PathBuf::from("/app")).is_ok());
        assert!(fs.symlink_metadata(&PathBuf::from("/app/a")).is_ok());
        assert!(fs.symlink_metadata(&PathBuf::from("/app/b")).is_ok());
        assert!(
            fs.symlink_metadata(&PathBuf::from("/app/a/data-a.txt"))
                .is_ok()
        );
        assert!(
            fs.symlink_metadata(&PathBuf::from("/app/b/data-b.txt"))
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_merge_preserves_nested_root_mounts_with_skip() {
        let primary = UnionFileSystem::new();
        let openssl = mem_fs::FileSystem::default();
        openssl.create_dir(Path::new("/certs")).unwrap();
        openssl
            .new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/certs/ca.pem"))
            .unwrap();
        primary
            .mount(
                "openssl".to_string(),
                Path::new("/openssl"),
                Box::new(openssl),
            )
            .unwrap();

        let injected = UnionFileSystem::new();
        let app = mem_fs::FileSystem::default();
        app.new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/index.php"))
            .unwrap();
        injected
            .mount("app".to_string(), Path::new("/app"), Box::new(app))
            .unwrap();

        let assets = mem_fs::FileSystem::default();
        assets.create_dir(Path::new("/css")).unwrap();
        assets
            .new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/css/site.css"))
            .unwrap();
        injected
            .mount(
                "assets".to_string(),
                Path::new("/opt/assets"),
                Box::new(assets),
            )
            .unwrap();

        primary.merge(&injected, UnionMergeMode::Skip).unwrap();

        let root_contents = read_dir_names(&primary, "/");
        assert!(root_contents.contains(&"app".to_string()));
        assert!(root_contents.contains(&"opt".to_string()));
        assert!(root_contents.contains(&"openssl".to_string()));
        assert!(primary.metadata(Path::new("/app/index.php")).is_ok());
        assert!(
            primary
                .metadata(Path::new("/opt/assets/css/site.css"))
                .is_ok()
        );
        assert!(primary.metadata(Path::new("/openssl/certs/ca.pem")).is_ok());
    }

    #[tokio::test]
    async fn test_nested_mount_under_non_mountable_leaf_is_supported() {
        let fs = UnionFileSystem::new();

        let top = MountlessFileSystem::default();
        top.create_dir(Path::new("/bin")).unwrap();
        top.new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/bin/tool"))
            .unwrap();

        let nested = mem_fs::FileSystem::default();
        nested.create_dir(Path::new("/css")).unwrap();
        nested
            .new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/css/site.css"))
            .unwrap();

        fs.mount("opt".to_string(), Path::new("/opt"), Box::new(top))
            .unwrap();
        fs.mount(
            "assets".to_string(),
            Path::new("/opt/assets"),
            Box::new(nested),
        )
        .unwrap();

        assert!(fs.metadata(Path::new("/opt/bin/tool")).is_ok());
        assert!(fs.metadata(Path::new("/opt/assets/css/site.css")).is_ok());
    }

    #[tokio::test]
    async fn test_parent_read_dir_merges_leaf_entries_with_child_mounts() {
        let fs = UnionFileSystem::new();

        let top = MountlessFileSystem::default();
        top.create_dir(Path::new("/bin")).unwrap();
        top.new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/bin/tool"))
            .unwrap();

        let nested = mem_fs::FileSystem::default();
        nested.create_dir(Path::new("/css")).unwrap();
        nested
            .new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/css/site.css"))
            .unwrap();

        fs.mount("opt".to_string(), Path::new("/opt"), Box::new(top))
            .unwrap();
        fs.mount(
            "assets".to_string(),
            Path::new("/opt/assets"),
            Box::new(nested),
        )
        .unwrap();

        let opt_contents = read_dir_names(&fs, "/opt");
        assert!(opt_contents.contains(&"bin".to_string()));
        assert!(opt_contents.contains(&"assets".to_string()));
    }

    #[tokio::test]
    async fn test_child_mount_shadows_same_named_parent_entry() {
        let fs = UnionFileSystem::new();

        let top = MountlessFileSystem::default();
        top.new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/assets"))
            .unwrap();

        let nested = mem_fs::FileSystem::default();
        nested.create_dir(Path::new("/css")).unwrap();
        nested
            .new_open_options()
            .write(true)
            .create_new(true)
            .open(Path::new("/css/site.css"))
            .unwrap();

        fs.mount("opt".to_string(), Path::new("/opt"), Box::new(top))
            .unwrap();
        fs.mount(
            "assets".to_string(),
            Path::new("/opt/assets"),
            Box::new(nested),
        )
        .unwrap();

        assert!(fs.metadata(Path::new("/opt/assets")).unwrap().is_dir());
        assert_eq!(
            read_dir_names(&fs, "/opt")
                .into_iter()
                .filter(|entry| entry == "assets")
                .count(),
            1,
        );
        assert!(fs.metadata(Path::new("/opt/assets/css/site.css")).is_ok());
    }

    #[tokio::test]
    async fn test_new_filesystem() {
        let fs = gen_filesystem();
        assert!(
            fs.read_dir(Path::new("/test_new_filesystem")).is_ok(),
            "hostfs can read root"
        );
        let mut file_write = fs
            .new_open_options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(Path::new("/test_new_filesystem/foo2.txt"))
            .unwrap();
        file_write.write_all(b"hello").await.unwrap();
        let _ = std::fs::remove_file("/test_new_filesystem/foo2.txt");
    }

    #[tokio::test]
    async fn test_create_dir() {
        let fs = gen_filesystem();

        assert_eq!(fs.create_dir(Path::new("/")), Ok(()));

        assert_eq!(fs.create_dir(Path::new("/test_create_dir")), Ok(()));

        assert_eq!(
            fs.create_dir(Path::new("/test_create_dir/foo")),
            Ok(()),
            "creating a directory",
        );

        let cur_dir = read_dir_names(&fs, "/test_create_dir");

        if !cur_dir.contains(&"foo".to_string()) {
            panic!("cur_dir does not contain foo: {cur_dir:#?}");
        }

        assert!(
            cur_dir.contains(&"foo".to_string()),
            "the root is updated and well-defined"
        );

        assert_eq!(
            fs.create_dir(Path::new("/test_create_dir/foo/bar")),
            Ok(()),
            "creating a sub-directory",
        );

        let foo_dir = read_dir_names(&fs, "/test_create_dir/foo");

        assert!(
            foo_dir.contains(&"bar".to_string()),
            "the foo directory is updated and well-defined"
        );

        let bar_dir = read_dir_names(&fs, "/test_create_dir/foo/bar");

        assert!(
            bar_dir.is_empty(),
            "the foo directory is updated and well-defined"
        );
        let _ = fs_extra::remove_items(&["/test_create_dir"]);
    }

    #[tokio::test]
    async fn test_remove_dir() {
        let fs = gen_filesystem();

        assert_eq!(
            fs.remove_dir(Path::new("/")),
            Err(FsError::PermissionDenied),
            "cannot remove the root directory",
        );

        assert_eq!(
            fs.remove_dir(Path::new("/foo")),
            Err(FsError::EntryNotFound),
            "cannot remove a directory that doesn't exist",
        );

        assert_eq!(fs.create_dir(Path::new("/test_remove_dir")), Ok(()));

        assert_eq!(
            fs.create_dir(Path::new("/test_remove_dir/foo")),
            Ok(()),
            "creating a directory",
        );

        assert_eq!(
            fs.create_dir(Path::new("/test_remove_dir/foo/bar")),
            Ok(()),
            "creating a sub-directory",
        );

        assert!(
            read_dir_names(&fs, "/test_remove_dir/foo").contains(&"bar".to_string()),
            "./foo/bar exists"
        );

        assert_eq!(
            fs.remove_dir(Path::new("/test_remove_dir/foo")),
            Err(FsError::DirectoryNotEmpty),
            "removing a directory that has children",
        );

        assert_eq!(
            fs.remove_dir(Path::new("/test_remove_dir/foo/bar")),
            Ok(()),
            "removing a sub-directory",
        );

        assert_eq!(
            fs.remove_dir(Path::new("/test_remove_dir/foo")),
            Ok(()),
            "removing a directory",
        );

        assert!(
            !read_dir_names(&fs, "/test_remove_dir").contains(&"foo".to_string()),
            "the foo directory still exists"
        );
    }

    fn read_dir_names(fs: &dyn crate::FileSystem, path: &str) -> Vec<String> {
        fs.read_dir(Path::new(path))
            .unwrap()
            .filter_map(|entry| Some(entry.ok()?.file_name().to_str()?.to_string()))
            .collect::<Vec<_>>()
    }

    #[tokio::test]
    async fn test_rename() {
        let fs = gen_filesystem();

        assert_eq!(
            fs.rename(Path::new("/"), Path::new("/bar")).await,
            Err(FsError::PermissionDenied),
            "renaming a directory that has no parent",
        );
        assert_eq!(
            fs.rename(Path::new("/foo"), Path::new("/")).await,
            Err(FsError::EntryNotFound),
            "renaming to a directory that has no parent",
        );

        assert_eq!(fs.create_dir(Path::new("/test_rename")), Ok(()));
        assert_eq!(fs.create_dir(Path::new("/test_rename/foo")), Ok(()));
        assert_eq!(fs.create_dir(Path::new("/test_rename/foo/qux")), Ok(()));

        assert_eq!(
            fs.rename(
                Path::new("/test_rename/foo"),
                Path::new("/test_rename/bar/baz")
            )
            .await,
            Err(FsError::EntryNotFound),
            "renaming to a directory that has parent that doesn't exist",
        );

        assert_eq!(fs.create_dir(Path::new("/test_rename/bar")), Ok(()));

        assert_eq!(
            fs.rename(Path::new("/test_rename/foo"), Path::new("/test_rename/bar"))
                .await,
            Ok(()),
            "renaming to a directory that has parent that exists",
        );

        assert!(
            fs.new_open_options()
                .write(true)
                .create_new(true)
                .open(Path::new("/test_rename/bar/hello1.txt"))
                .is_ok(),
            "creating a new file (`hello1.txt`)",
        );
        assert!(
            fs.new_open_options()
                .write(true)
                .create_new(true)
                .open(Path::new("/test_rename/bar/hello2.txt"))
                .is_ok(),
            "creating a new file (`hello2.txt`)",
        );

        let cur_dir = read_dir_names(&fs, "/test_rename");

        assert!(
            !cur_dir.contains(&"foo".to_string()),
            "the foo directory still exists"
        );

        assert!(
            cur_dir.contains(&"bar".to_string()),
            "the bar directory still exists"
        );

        let bar_dir = read_dir_names(&fs, "/test_rename/bar");

        if !bar_dir.contains(&"qux".to_string()) {
            println!("qux does not exist: {bar_dir:?}")
        }

        let qux_dir = read_dir_names(&fs, "/test_rename/bar/qux");

        assert!(qux_dir.is_empty(), "the qux directory is empty");

        assert!(
            read_dir_names(&fs, "/test_rename/bar").contains(&"hello1.txt".to_string()),
            "the /bar/hello1.txt file exists"
        );

        assert!(
            read_dir_names(&fs, "/test_rename/bar").contains(&"hello2.txt".to_string()),
            "the /bar/hello2.txt file exists"
        );

        assert_eq!(
            fs.create_dir(Path::new("/test_rename/foo")),
            Ok(()),
            "create ./foo again",
        );

        assert_eq!(
            fs.rename(
                Path::new("/test_rename/bar/hello2.txt"),
                Path::new("/test_rename/foo/world2.txt")
            )
            .await,
            Ok(()),
            "renaming (and moving) a file",
        );

        assert_eq!(
            fs.rename(
                Path::new("/test_rename/foo"),
                Path::new("/test_rename/bar/baz")
            )
            .await,
            Ok(()),
            "renaming a directory",
        );

        assert_eq!(
            fs.rename(
                Path::new("/test_rename/bar/hello1.txt"),
                Path::new("/test_rename/bar/world1.txt")
            )
            .await,
            Ok(()),
            "renaming a file (in the same directory)",
        );

        assert!(
            read_dir_names(&fs, "/test_rename").contains(&"bar".to_string()),
            "./bar exists"
        );

        assert!(
            read_dir_names(&fs, "/test_rename/bar").contains(&"baz".to_string()),
            "/bar/baz exists"
        );
        assert!(
            !read_dir_names(&fs, "/test_rename").contains(&"foo".to_string()),
            "foo does not exist anymore"
        );
        assert!(
            read_dir_names(&fs, "/test_rename/bar/baz").contains(&"world2.txt".to_string()),
            "/bar/baz/world2.txt exists"
        );
        assert!(
            read_dir_names(&fs, "/test_rename/bar").contains(&"world1.txt".to_string()),
            "/bar/world1.txt (ex hello1.txt) exists"
        );
        assert!(
            !read_dir_names(&fs, "/test_rename/bar").contains(&"hello1.txt".to_string()),
            "hello1.txt was moved"
        );
        assert!(
            !read_dir_names(&fs, "/test_rename/bar").contains(&"hello2.txt".to_string()),
            "hello2.txt was moved"
        );
        assert!(
            read_dir_names(&fs, "/test_rename/bar/baz").contains(&"world2.txt".to_string()),
            "world2.txt was moved to the correct place"
        );

        let _ = fs_extra::remove_items(&["/test_rename"]);
    }

    #[tokio::test]
    async fn test_metadata() {
        use std::thread::sleep;
        use std::time::Duration;

        let fs = gen_filesystem();

        let root_metadata = fs.metadata(Path::new("/test_metadata")).unwrap();

        assert!(root_metadata.ft.dir);
        assert_eq!(root_metadata.accessed, root_metadata.created);
        assert_eq!(root_metadata.modified, root_metadata.created);
        assert!(root_metadata.modified > 0);

        assert_eq!(fs.create_dir(Path::new("/test_metadata/foo")), Ok(()));

        let foo_metadata = fs.metadata(Path::new("/test_metadata/foo"));
        assert!(foo_metadata.is_ok());
        let foo_metadata = foo_metadata.unwrap();

        assert!(foo_metadata.ft.dir);
        assert!(foo_metadata.accessed == foo_metadata.created);
        assert!(foo_metadata.modified == foo_metadata.created);
        assert!(foo_metadata.modified > 0);

        sleep(Duration::from_secs(3));

        assert_eq!(
            fs.rename(
                Path::new("/test_metadata/foo"),
                Path::new("/test_metadata/bar")
            )
            .await,
            Ok(())
        );

        let bar_metadata = fs.metadata(Path::new("/test_metadata/bar")).unwrap();
        assert!(bar_metadata.ft.dir);
        assert!(bar_metadata.accessed == foo_metadata.accessed);
        assert!(bar_metadata.created == foo_metadata.created);
        assert!(bar_metadata.modified > foo_metadata.modified);

        let root_metadata = fs.metadata(Path::new("/test_metadata/bar")).unwrap();
        assert!(
            root_metadata.modified > foo_metadata.modified,
            "the parent modified time was updated"
        );

        let _ = fs_extra::remove_items(&["/test_metadata"]);
    }

    #[tokio::test]
    async fn test_remove_file() {
        let fs = gen_filesystem();

        assert!(
            fs.new_open_options()
                .write(true)
                .create_new(true)
                .open(Path::new("/test_remove_file/foo.txt"))
                .is_ok(),
            "creating a new file",
        );

        assert!(read_dir_names(&fs, "/test_remove_file").contains(&"foo.txt".to_string()));

        assert_eq!(
            fs.remove_file(Path::new("/test_remove_file/foo.txt")),
            Ok(()),
            "removing a file that exists",
        );

        assert!(!read_dir_names(&fs, "/test_remove_file").contains(&"foo.txt".to_string()));

        assert_eq!(
            fs.remove_file(Path::new("/test_remove_file/foo.txt")),
            Err(FsError::EntryNotFound),
            "removing a file that doesn't exists",
        );

        let _ = fs_extra::remove_items(&["./test_remove_file"]);
    }

    #[tokio::test]
    async fn test_readdir() {
        let fs = gen_filesystem();

        assert_eq!(
            fs.create_dir(Path::new("/test_readdir/foo")),
            Ok(()),
            "creating `foo`"
        );
        assert_eq!(
            fs.create_dir(Path::new("/test_readdir/foo/sub")),
            Ok(()),
            "creating `sub`"
        );
        assert_eq!(
            fs.create_dir(Path::new("/test_readdir/bar")),
            Ok(()),
            "creating `bar`"
        );
        assert_eq!(
            fs.create_dir(Path::new("/test_readdir/baz")),
            Ok(()),
            "creating `bar`"
        );
        assert!(
            fs.new_open_options()
                .write(true)
                .create_new(true)
                .open(Path::new("/test_readdir/a.txt"))
                .is_ok(),
            "creating `a.txt`",
        );
        assert!(
            fs.new_open_options()
                .write(true)
                .create_new(true)
                .open(Path::new("/test_readdir/b.txt"))
                .is_ok(),
            "creating `b.txt`",
        );

        println!("fs: {fs:?}");

        let readdir = fs.read_dir(Path::new("/test_readdir"));

        assert!(readdir.is_ok(), "reading the directory `/test_readdir/`");

        let mut readdir = readdir.unwrap();

        let next = readdir.next().unwrap().unwrap();
        assert!(next.path.ends_with("foo"), "checking entry #1");
        println!("entry 1: {next:#?}");
        assert!(next.file_type().unwrap().is_dir(), "checking entry #1");

        let next = readdir.next().unwrap().unwrap();
        assert!(next.path.ends_with("bar"), "checking entry #2");
        assert!(next.file_type().unwrap().is_dir(), "checking entry #2");

        let next = readdir.next().unwrap().unwrap();
        assert!(next.path.ends_with("baz"), "checking entry #3");
        assert!(next.file_type().unwrap().is_dir(), "checking entry #3");

        let next = readdir.next().unwrap().unwrap();
        assert!(next.path.ends_with("a.txt"), "checking entry #2");
        assert!(next.file_type().unwrap().is_file(), "checking entry #4");

        let next = readdir.next().unwrap().unwrap();
        assert!(next.path.ends_with("b.txt"), "checking entry #2");
        assert!(next.file_type().unwrap().is_file(), "checking entry #5");

        if let Some(s) = readdir.next() {
            panic!("next: {s:?}");
        }

        let _ = fs_extra::remove_items(&["./test_readdir"]);
    }

    /*
    #[tokio::test]
    async fn test_canonicalize() {
        let fs = gen_filesystem();

        let root_dir = env!("CARGO_MANIFEST_DIR");

        let _ = fs_extra::remove_items(&["./test_canonicalize"]);

        assert_eq!(
            fs.create_dir(Path::new("./test_canonicalize")),
            Ok(()),
            "creating `test_canonicalize`"
        );

        assert_eq!(
            fs.create_dir(Path::new("./test_canonicalize/foo")),
            Ok(()),
            "creating `foo`"
        );
        assert_eq!(
            fs.create_dir(Path::new("./test_canonicalize/foo/bar")),
            Ok(()),
            "creating `bar`"
        );
        assert_eq!(
            fs.create_dir(Path::new("./test_canonicalize/foo/bar/baz")),
            Ok(()),
            "creating `baz`",
        );
        assert_eq!(
            fs.create_dir(Path::new("./test_canonicalize/foo/bar/baz/qux")),
            Ok(()),
            "creating `qux`",
        );
        assert!(
            matches!(
                fs.new_open_options()
                    .write(true)
                    .create_new(true)
                    .open(Path::new("./test_canonicalize/foo/bar/baz/qux/hello.txt")),
                Ok(_)
            ),
            "creating `hello.txt`",
        );

        assert_eq!(
            fs.canonicalize(Path::new("./test_canonicalize")),
            Ok(Path::new(&format!("{root_dir}/test_canonicalize")).to_path_buf()),
            "canonicalizing `/`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("foo")),
            Err(FsError::InvalidInput),
            "canonicalizing `foo`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("./test_canonicalize/././././foo/")),
            Ok(Path::new(&format!("{root_dir}/test_canonicalize/foo")).to_path_buf()),
            "canonicalizing `/././././foo/`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("./test_canonicalize/foo/bar//")),
            Ok(Path::new(&format!("{root_dir}/test_canonicalize/foo/bar")).to_path_buf()),
            "canonicalizing `/foo/bar//`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("./test_canonicalize/foo/bar/../bar")),
            Ok(Path::new(&format!("{root_dir}/test_canonicalize/foo/bar")).to_path_buf()),
            "canonicalizing `/foo/bar/../bar`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("./test_canonicalize/foo/bar/../..")),
            Ok(Path::new(&format!("{root_dir}/test_canonicalize")).to_path_buf()),
            "canonicalizing `/foo/bar/../..`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("/foo/bar/../../..")),
            Err(FsError::InvalidInput),
            "canonicalizing `/foo/bar/../../..`",
        );
        assert_eq!(
            fs.canonicalize(Path::new("C:/foo/")),
            Err(FsError::InvalidInput),
            "canonicalizing `C:/foo/`",
        );
        assert_eq!(
            fs.canonicalize(Path::new(
                "./test_canonicalize/foo/./../foo/bar/../../foo/bar/./baz/./../baz/qux/../../baz/./qux/hello.txt"
            )),
            Ok(Path::new(&format!("{root_dir}/test_canonicalize/foo/bar/baz/qux/hello.txt")).to_path_buf()),
            "canonicalizing a crazily stupid path name",
        );

        let _ = fs_extra::remove_items(&["./test_canonicalize"]);
    }
    */
}
