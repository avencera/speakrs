//! Hugging Face Hub downloads into the standard hub cache
//!
//! The endpoint, token and cache resolve the way `huggingface_hub` resolves them, so Python
//! tools and this crate share one cache. Downloads use a small blocking `ureq` client instead
//! of hf-hub 1.0, whose async stack (tokio, reqwest, hf-xet) is far more than fetching a few
//! model files needs
//!
//! speakrs, scriptrs and sam3rs carry the same module; change them together

use std::env;
use std::path::PathBuf;

/// Value of the environment variable `name`, treating an empty value as unset
fn env_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

/// `$HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`
fn hf_home() -> PathBuf {
    if let Some(home) = env_var("HF_HOME") {
        return PathBuf::from(home);
    }

    let cache = env_var("XDG_CACHE_HOME").map_or_else(
        || env::home_dir().unwrap_or_else(env::temp_dir).join(".cache"),
        PathBuf::from,
    );
    cache.join("huggingface")
}

/// The Hugging Face hub cache that downloads use unless given another directory
///
/// `$HF_HUB_CACHE`, else `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, where `HF_HOME`
/// defaults to `$XDG_CACHE_HOME/huggingface`, else `~/.cache/huggingface`
pub fn default_hub_cache_dir() -> PathBuf {
    env_var("HF_HUB_CACHE")
        .or_else(|| env_var("HUGGINGFACE_HUB_CACHE"))
        .map_or_else(|| hf_home().join("hub"), PathBuf::from)
}

#[cfg(feature = "online")]
pub use download::{Hub, HubError, HubRepo};

#[cfg(feature = "online")]
mod download {
    use std::fs;
    use std::io;
    use std::path::{Component, Path, PathBuf};

    use ureq::Agent;
    use ureq::http::Response;

    use super::{default_hub_cache_dir, env_var, hf_home};

    const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

    // the Hub redirects renamed repos to their new name, so a few hops are legitimate
    const MAX_REDIRECTS: usize = 5;

    /// Why a Hub download failed
    #[derive(Debug, thiserror::Error)]
    #[non_exhaustive]
    pub enum HubError {
        /// The Hub answered with an error status: 401 without a valid token, 403 when the
        /// token's account cannot access the repo, 404 for a missing repo, revision or file
        #[error("Hugging Face answered HTTP {status} for {url}")]
        Status {
            /// HTTP status code
            status: u16,
            /// Requested URL
            url: String,
        },
        /// The request failed before the Hub answered
        #[error("request {url}: {error}")]
        Request {
            /// Requested URL
            url: String,
            /// Transport error
            error: Box<ureq::Error>,
        },
        /// The Hub's answer lacks a header that a download needs
        #[error("Hugging Face's answer for {url} has no {header} header")]
        MissingHeader {
            /// Requested URL
            url: String,
            /// Name of the missing header
            header: &'static str,
        },
        /// The Hub's answer has a header value that cannot name a cache entry, such as a
        /// commit that is not a hash or an etag with path separators
        #[error("Hugging Face's answer for {url} has an invalid {header} header: {value:?}")]
        InvalidHeader {
            /// Requested URL
            url: String,
            /// Name of the invalid header
            header: &'static str,
            /// Value the Hub sent
            value: String,
        },
        /// The requested file is not a relative path inside the repo
        #[error("{file:?} is not a file path inside the repo")]
        InvalidFile {
            /// Requested file
            file: String,
        },
        /// The Hub kept redirecting
        #[error("Hugging Face redirected {url} more than {MAX_REDIRECTS} times")]
        TooManyRedirects {
            /// First requested URL
            url: String,
        },
        /// Reading the token, downloading into the cache or linking a snapshot file failed
        #[error("{}: {error}", path.display())]
        Io {
            /// File the operation was on
            path: PathBuf,
            /// I/O error
            error: io::Error,
        },
    }

    impl HubError {
        /// The HTTP status of a [`HubError::Status`]
        pub fn status(&self) -> Option<u16> {
            match self {
                Self::Status { status, .. } => Some(*status),
                _ => None,
            }
        }

        fn io(path: &Path, error: io::Error) -> Self {
            Self::Io {
                path: path.to_path_buf(),
                error,
            }
        }
    }

    /// Blocking Hub client, configured from the environment like `huggingface_hub`
    #[derive(Clone, Debug)]
    pub struct Hub {
        agent: Agent,
        endpoint: String,
        token: Option<String>,
        cache_dir: PathBuf,
    }

    impl Hub {
        /// Client for the cache at [`default_hub_cache_dir`]
        ///
        /// The endpoint is `$HF_ENDPOINT`, else `https://huggingface.co`. The token is
        /// `$HF_TOKEN`, else the file at `$HF_TOKEN_PATH`, else `$HF_HOME/token`, where
        /// `hf auth login` saves it
        pub fn from_env() -> Result<Self, HubError> {
            Self::with_cache_dir(default_hub_cache_dir())
        }

        /// Client for the cache at `cache_dir`, otherwise configured like [`Hub::from_env`]
        pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self, HubError> {
            let agent = Agent::config_builder()
                // error answers carry what callers report, such as a gated repo's 401
                .http_status_as_error(false)
                // redirects are followed by hand so the token never leaves the Hub
                .max_redirects(0)
                .user_agent(concat!(
                    env!("CARGO_PKG_NAME"),
                    "/",
                    env!("CARGO_PKG_VERSION")
                ))
                .build()
                .new_agent();
            let endpoint = env_var("HF_ENDPOINT").map_or_else(
                || DEFAULT_ENDPOINT.to_owned(),
                |e| e.trim_end_matches('/').to_owned(),
            );

            Ok(Self {
                agent,
                endpoint,
                token: read_token()?,
                cache_dir: cache_dir.into(),
            })
        }

        /// Files of the model repo `id` (`owner/name`) at `revision`, which is a branch, a
        /// tag or a commit hash
        pub fn model(&self, id: impl Into<String>, revision: impl Into<String>) -> HubRepo {
            HubRepo {
                hub: self.clone(),
                id: id.into(),
                revision: revision.into(),
            }
        }

        fn get(&self, url: &str, with_token: bool) -> Result<Response<ureq::Body>, HubError> {
            let mut request = self.agent.get(url);
            if with_token && let Some(token) = &self.token {
                request = request.header("Authorization", format!("Bearer {token}"));
            }

            request.call().map_err(|error| HubError::Request {
                url: url.to_owned(),
                error: Box::new(error),
            })
        }

        /// Asks for the first byte of `url` to learn the file's commit and blob etag
        ///
        /// Follows the Hub's own relative redirects with the token and stops at an absolute
        /// one, which points at the storage that serves the file
        fn metadata(&self, url: &str) -> Result<Metadata, HubError> {
            let mut current = url.to_owned();
            for _ in 0..=MAX_REDIRECTS {
                let mut request = self.agent.get(&current).header("Range", "bytes=0-0");
                if let Some(token) = &self.token {
                    request = request.header("Authorization", format!("Bearer {token}"));
                }

                let response = request.call().map_err(|error| HubError::Request {
                    url: current.clone(),
                    error: Box::new(error),
                })?;
                let status = response.status();
                let location = if status.is_redirection() {
                    let location = header(&response, &current, "location")?;
                    if location.starts_with('/') {
                        current = format!("{}{location}", origin(&current));
                        continue;
                    }
                    Download::Storage(location)
                } else if status.is_success() {
                    Download::Hub(current.clone())
                } else {
                    return Err(HubError::Status {
                        status: status.as_u16(),
                        url: current,
                    });
                };

                let etag = match header(&response, &current, "x-linked-etag") {
                    Ok(etag) => etag,
                    Err(_) => header(&response, &current, "etag")?,
                };
                // weak validators and quotes are not part of the blob name
                let etag = etag.trim_start_matches("W/").replace('"', "");
                let commit = header(&response, &current, "x-repo-commit")?;
                // both become cache paths, so a hostile endpoint must not reach outside it
                let invalid = |header, value: String| HubError::InvalidHeader {
                    url: current.clone(),
                    header,
                    value,
                };
                if !is_blob_name(&etag) {
                    return Err(invalid("etag", etag));
                }
                if !is_commit_hash(&commit) {
                    return Err(invalid("x-repo-commit", commit));
                }

                return Ok(Metadata {
                    commit,
                    etag,
                    location,
                });
            }

            Err(HubError::TooManyRedirects {
                url: url.to_owned(),
            })
        }

        /// Streams the file into `blob` through a temporary file, so an interrupted download
        /// never leaves a partial blob behind
        fn fetch(&self, location: &Download, blob: &Path) -> Result<(), HubError> {
            let response = match location {
                Download::Hub(url) => self.get(url, true)?,
                // storage URLs are presigned, so they get no token, and may redirect again
                Download::Storage(url) => self
                    .agent
                    .get(url)
                    .config()
                    .max_redirects(MAX_REDIRECTS as u32)
                    .build()
                    .call()
                    .map_err(|error| HubError::Request {
                        url: url.clone(),
                        error: Box::new(error),
                    })?,
            };
            if !response.status().is_success() {
                return Err(HubError::Status {
                    status: response.status().as_u16(),
                    url: location.url().to_owned(),
                });
            }

            let blobs = blob.parent().expect("blob paths sit in blobs/");
            fs::create_dir_all(blobs).map_err(|error| HubError::io(blobs, error))?;
            // the process id keeps two processes fetching the same blob out of each other's way
            let partial = blob.with_extension(format!("{}.incomplete", std::process::id()));
            let mut file =
                fs::File::create(&partial).map_err(|error| HubError::io(&partial, error))?;
            let copied = io::copy(&mut response.into_body().into_reader(), &mut file);
            drop(file);
            if let Err(error) = copied {
                let _ = fs::remove_file(&partial);
                return Err(HubError::io(&partial, error));
            }

            fs::rename(&partial, blob).map_err(|error| HubError::io(blob, error))
        }
    }

    /// Where the bytes of a file come from
    #[derive(Debug)]
    enum Download {
        /// The Hub serves the file itself, as it does for small files kept in git
        Hub(String),
        /// The Hub redirected to object storage, as it does for LFS and Xet files
        Storage(String),
    }

    impl Download {
        fn url(&self) -> &str {
            match self {
                Self::Hub(url) | Self::Storage(url) => url,
            }
        }
    }

    #[derive(Debug)]
    struct Metadata {
        commit: String,
        etag: String,
        location: Download,
    }

    /// One model repo at one revision, see [`Hub::model`]
    #[derive(Clone, Debug)]
    pub struct HubRepo {
        hub: Hub,
        id: String,
        revision: String,
    }

    impl HubRepo {
        /// The cached copy of `file`, found without a network request
        ///
        /// A commit hash names its snapshot directly; a branch or tag goes through the commit
        /// that its `refs/` entry recorded at the last download
        pub fn cached(&self, file: &str) -> Option<PathBuf> {
            if !is_repo_file(file) {
                return None;
            }

            let repo_dir = self.repo_dir();
            let commit = if is_commit_hash(&self.revision) {
                self.revision.clone()
            } else {
                let reference = repo_dir.join("refs").join(&self.revision);
                let commit = fs::read_to_string(reference).ok()?.trim().to_owned();
                is_commit_hash(&commit).then_some(commit)?
            };

            let path = repo_dir.join("snapshots").join(commit).join(file);
            // a link whose blob was deleted is not a cached file
            path.is_file().then_some(path)
        }

        /// The cached copy of `file`, downloaded first when the cache lacks it
        ///
        /// A cached file is used without asking the Hub whether a branch moved, so files
        /// already downloaded load offline and without a token
        pub fn get(&self, file: &str) -> Result<PathBuf, HubError> {
            match self.cached(file) {
                Some(path) => Ok(path),
                None => self.download(file),
            }
        }

        /// Downloads `file` into the cache, even when a copy is cached, and returns its
        /// snapshot path
        pub fn download(&self, file: &str) -> Result<PathBuf, HubError> {
            if !is_repo_file(file) {
                return Err(HubError::InvalidFile {
                    file: file.to_owned(),
                });
            }

            let url = format!(
                "{}/{}/resolve/{}/{file}",
                self.hub.endpoint,
                self.id,
                // a revision such as `refs/pr/1` is one path segment
                self.revision.replace('/', "%2F"),
            );
            let metadata = self.hub.metadata(&url)?;
            // a pinned commit must land in its own snapshot, not one the Hub picks
            if is_commit_hash(&self.revision) && metadata.commit != self.revision {
                return Err(HubError::InvalidHeader {
                    url,
                    header: "x-repo-commit",
                    value: metadata.commit,
                });
            }

            let repo_dir = self.repo_dir();
            let blob = repo_dir.join("blobs").join(&metadata.etag);
            if !blob.is_file() {
                self.hub.fetch(&metadata.location, &blob)?;
            }

            let snapshot_file = repo_dir.join("snapshots").join(&metadata.commit).join(file);
            link_blob(&blob, &snapshot_file, file)?;
            if metadata.commit != self.revision {
                let reference = repo_dir.join("refs").join(&self.revision);
                let refs = reference.parent().expect("refs sit in refs/");
                fs::create_dir_all(refs).map_err(|error| HubError::io(refs, error))?;
                fs::write(&reference, &metadata.commit)
                    .map_err(|error| HubError::io(&reference, error))?;
            }

            Ok(snapshot_file)
        }

        fn repo_dir(&self) -> PathBuf {
            self.hub
                .cache_dir
                .join(format!("models--{}", self.id.replace('/', "--")))
        }
    }

    fn read_token() -> Result<Option<String>, HubError> {
        if let Some(token) = env_var("HF_TOKEN") {
            return Ok(Some(token));
        }

        let path = env_var("HF_TOKEN_PATH").map_or_else(|| hf_home().join("token"), PathBuf::from);
        match fs::read_to_string(&path) {
            Ok(token) => Ok(Some(token.trim().to_owned()).filter(|token| !token.is_empty())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(HubError::io(&path, error)),
        }
    }

    fn header(
        response: &Response<ureq::Body>,
        url: &str,
        name: &'static str,
    ) -> Result<String, HubError> {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
            .ok_or_else(|| HubError::MissingHeader {
                url: url.to_owned(),
                header: name,
            })
    }

    /// `scheme://host[:port]` of `url`
    fn origin(url: &str) -> &str {
        let host_start = url.find("://").map_or(0, |index| index + 3);
        url[host_start..]
            .find('/')
            .map_or(url, |index| &url[..host_start + index])
    }

    fn is_commit_hash(revision: &str) -> bool {
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    /// Hub etags are git or SHA-256 hashes, so anything else cannot be a blob name
    fn is_blob_name(etag: &str) -> bool {
        !etag.is_empty()
            && etag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    }

    /// A relative path that stays inside the snapshot, such as `dir/model.bin`
    fn is_repo_file(file: &str) -> bool {
        !file.is_empty()
            && Path::new(file)
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
    }

    /// Points `snapshot_file` at `blob` with a relative link, as `huggingface_hub` does, so
    /// the cache stays valid when it moves
    fn link_blob(blob: &Path, snapshot_file: &Path, file: &str) -> Result<(), HubError> {
        if snapshot_file.is_file() {
            return Ok(());
        }

        let parent = snapshot_file
            .parent()
            .expect("snapshot files sit in snapshots/");
        fs::create_dir_all(parent).map_err(|error| HubError::io(parent, error))?;
        // a dangling link from a deleted blob would block the new one
        if snapshot_file.symlink_metadata().is_ok() {
            fs::remove_file(snapshot_file).map_err(|error| HubError::io(snapshot_file, error))?;
        }

        // up from the file's directory to the snapshot, then out of snapshots/<commit>/
        let depth = Path::new(file).components().count() + 1;
        let mut target = PathBuf::new();
        for _ in 0..depth {
            target.push("..");
        }
        target.push("blobs");
        target.push(blob.file_name().expect("blob paths end in the etag"));

        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&target, snapshot_file);
        // symlinks need extra privileges on Windows, so the snapshot gets a copy there
        #[cfg(not(unix))]
        let linked = fs::copy(blob, snapshot_file).map(drop);
        match linked {
            Ok(()) => Ok(()),
            // another process linked it first
            Err(_) if snapshot_file.is_file() => Ok(()),
            Err(error) => Err(HubError::io(snapshot_file, error)),
        }
    }

    #[cfg(test)]
    mod tests {
        use std::io::{BufRead, BufReader, Write};
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};
        use std::thread;

        use super::*;

        const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
        const BODY: &[u8] = b"weights";

        /// Path and `Authorization` header of each request the test server received
        type Requests = Arc<Mutex<Vec<(String, Option<String>)>>>;

        /// Serves one repo file: the Hub answers with a redirect to storage, which serves
        /// the bytes, and every request's path and token are recorded
        fn serve() -> (String, Requests) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let recorded = Arc::clone(&requests);
            let storage = format!("{origin}/storage/blob");
            thread::spawn(move || {
                for stream in listener.incoming() {
                    let mut stream = stream.unwrap();
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let path = line.split_whitespace().nth(1).unwrap().to_owned();
                    let mut auth = None;
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).unwrap();
                        if header.trim().is_empty() {
                            break;
                        }
                        if let Some(value) = header.strip_prefix("authorization: ") {
                            auth = Some(value.trim().to_owned());
                        }
                    }
                    recorded.lock().unwrap().push((path.clone(), auth));

                    let response = if path.starts_with("/storage/") {
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                            BODY.len(),
                            String::from_utf8_lossy(BODY)
                        )
                    } else if path.starts_with("/old/") {
                        // a renamed repo redirects relative to the Hub
                        let renamed = path.replacen("/old/", "/owner/", 1);
                        format!(
                            "HTTP/1.1 307 Temporary Redirect\r\nlocation: {renamed}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        )
                    } else if path.contains("/missing") {
                        "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                            .to_owned()
                    } else {
                        format!(
                            "HTTP/1.1 302 Found\r\nlocation: {storage}\r\nx-repo-commit: {COMMIT}\r\nx-linked-etag: \"abc123\"\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                        )
                    };
                    stream.write_all(response.as_bytes()).unwrap();
                }
            });

            (origin, requests)
        }

        fn hub(origin: &str, cache_dir: &Path) -> Hub {
            let mut hub = Hub::with_cache_dir(cache_dir).unwrap();
            hub.endpoint = origin.to_owned();
            hub.token = Some("secret".to_owned());
            hub
        }

        #[test]
        fn downloads_into_hub_cache_layout_and_keeps_token_on_the_hub() {
            let (origin, requests) = serve();
            let cache = tempfile::tempdir().unwrap();
            let repo = hub(&origin, cache.path()).model("old/model", "main");

            let path = repo.get("nested/model.bin").unwrap();

            let repo_dir = cache.path().join("models--old--model");
            assert_eq!(
                path,
                repo_dir
                    .join("snapshots")
                    .join(COMMIT)
                    .join("nested/model.bin")
            );
            assert_eq!(fs::read(&path).unwrap(), BODY);
            assert_eq!(
                fs::read_link(&path).unwrap(),
                Path::new("../../../blobs/abc123")
            );
            assert_eq!(
                fs::read_to_string(repo_dir.join("refs/main")).unwrap(),
                COMMIT
            );

            let requests = requests.lock().unwrap().clone();
            let paths: Vec<_> = requests.iter().map(|(path, _)| path.as_str()).collect();
            assert_eq!(
                paths,
                [
                    "/old/model/resolve/main/nested/model.bin",
                    "/owner/model/resolve/main/nested/model.bin",
                    "/storage/blob",
                ]
            );
            let bearer = Some("Bearer secret".to_owned());
            assert_eq!(requests[0].1, bearer);
            assert_eq!(requests[1].1, bearer);
            assert_eq!(requests[2].1, None);
        }

        #[test]
        fn cached_files_resolve_without_requests() {
            let (origin, requests) = serve();
            let cache = tempfile::tempdir().unwrap();
            let hub = hub(&origin, cache.path());
            hub.model("owner/model", "main").get("model.bin").unwrap();
            let before = requests.lock().unwrap().len();

            // through the branch's ref, and by the commit it recorded
            let by_branch = hub.model("owner/model", "main").get("model.bin").unwrap();
            let by_commit = hub.model("owner/model", COMMIT).get("model.bin").unwrap();

            assert_eq!(by_branch, by_commit);
            assert_eq!(requests.lock().unwrap().len(), before);
        }

        #[test]
        fn rejects_names_that_leave_the_cache() {
            let (origin, requests) = serve();
            let cache = tempfile::tempdir().unwrap();
            let hub = hub(&origin, cache.path());

            let escaping = hub.model("owner/model", "main").get("../../outside.bin");
            // the server answers with commit COMMIT, not the pinned one
            let other_commit = "fedcba9876543210fedcba9876543210fedcba98";
            let mismatched = hub.model("owner/model", other_commit).get("model.bin");

            assert!(matches!(escaping, Err(HubError::InvalidFile { .. })));
            assert!(matches!(
                mismatched,
                Err(HubError::InvalidHeader {
                    header: "x-repo-commit",
                    ..
                })
            ));
            assert_eq!(requests.lock().unwrap().len(), 1);
            assert!(!cache.path().join("models--owner--model/snapshots").exists());
        }

        #[test]
        fn error_status_reaches_the_caller() {
            let (origin, _) = serve();
            let cache = tempfile::tempdir().unwrap();
            let repo = hub(&origin, cache.path()).model("owner/missing", "main");

            let error = repo.get("model.bin").unwrap_err();

            assert_eq!(error.status(), Some(404));
            assert!(!cache.path().join("models--owner--missing").exists());
        }
    }
}
