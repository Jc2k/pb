use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
#[cfg(target_os = "macos")]
use image::{ImageFormat, ImageReader};
#[cfg(target_os = "macos")]
use objc2::rc::{Retained, autoreleasepool};
#[cfg(target_os = "macos")]
use objc2_foundation::{NSData, NSError, NSString};
#[cfg(target_os = "macos")]
use objc2_open_directory::{
    ODAttributeType, ODNode, ODRecord, ODRecordType, kODAttributeTypeFullName,
    kODAttributeTypeJPEGPhoto, kODAttributeTypePicture, kODNodeTypeLocalNodes, kODRecordTypeUsers,
};
use serde::Serialize;
#[cfg(target_os = "macos")]
use std::{ffi::CStr, io::Cursor, path::PathBuf};
use thiserror::Error;
use tokio::task;

pub async fn avatar_png() -> Response {
    match task::spawn_blocking(load_current_process_user_avatar_png).await {
        Ok(Ok(png)) => {
            let headers = [
                (header::CONTENT_TYPE, HeaderValue::from_static("image/png")),
                (
                    header::CACHE_CONTROL,
                    HeaderValue::from_static("private, max-age=60"),
                ),
            ];

            (headers, png).into_response()
        }
        Ok(Err(err)) => {
            tracing::debug!("could not load macOS user avatar: {err}");
            StatusCode::NOT_FOUND.into_response()
        }
        Err(err) => {
            tracing::error!("avatar task failed: {err}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

pub async fn user_info() -> Response {
    match task::spawn_blocking(load_current_process_user_info).await {
        Ok(Ok(info)) => Json(info).into_response(),
        Ok(Err(err)) => {
            tracing::debug!("could not load macOS user info: {err}");
            StatusCode::NOT_FOUND.into_response()
        }
        Err(err) => {
            tracing::error!("user info task failed: {err}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct UserInfo {
    pub username: String,
    pub real_name: Option<String>,
}

#[cfg(not(target_os = "macos"))]
fn load_current_process_user_avatar_png() -> Result<Vec<u8>, MacOsUserError> {
    Err(MacOsUserError::UnsupportedPlatform)
}

#[cfg(not(target_os = "macos"))]
fn load_current_process_user_info() -> Result<UserInfo, MacOsUserError> {
    Err(MacOsUserError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn load_current_process_user_avatar_png() -> Result<Vec<u8>, MacOsUserError> {
    let username = current_effective_username()?;

    autoreleasepool(|_| {
        let record = open_directory_user_record(&username)?;
        let source_bytes = read_avatar_source_bytes(&record)?;
        encode_png(&source_bytes)
    })
}

#[cfg(target_os = "macos")]
fn load_current_process_user_info() -> Result<UserInfo, MacOsUserError> {
    let username = current_effective_username()?;

    autoreleasepool(|_| {
        let record = open_directory_user_record(&username)?;

        let real_name = od_first_string_attr(&record, od_attr_full_name()?)?
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());

        Ok(UserInfo {
            username,
            real_name,
        })
    })
}

#[cfg(target_os = "macos")]
fn current_effective_username() -> Result<String, MacOsUserError> {
    unsafe {
        let pw = libc::getpwuid(libc::geteuid());

        if pw.is_null() {
            return Err(MacOsUserError::NoCurrentUser);
        }

        let username = CStr::from_ptr((*pw).pw_name).to_string_lossy().into_owned();

        if username.is_empty() {
            Err(MacOsUserError::NoCurrentUser)
        } else {
            Ok(username)
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn current_username() -> Option<String> {
    current_effective_username().ok()
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn current_username() -> Option<String> {
    None
}

#[cfg(target_os = "macos")]
fn open_directory_user_record(username: &str) -> Result<Retained<ODRecord>, MacOsUserError> {
    unsafe {
        let mut err: Option<Retained<NSError>> = None;

        let node = ODNode::nodeWithSession_type_error(None, kODNodeTypeLocalNodes, Some(&mut err))
            .ok_or_else(|| MacOsUserError::OpenDirectory(ns_error_string(err.take())))?;

        let username = NSString::from_str(username);

        err = None;

        node.recordWithRecordType_name_attributes_error(
            Some(od_record_type_users()?),
            Some(&username),
            None,
            Some(&mut err),
        )
        .ok_or_else(|| MacOsUserError::OpenDirectory(ns_error_string(err.take())))
    }
}

#[cfg(target_os = "macos")]
fn read_avatar_source_bytes(record: &ODRecord) -> Result<Vec<u8>, MacOsUserError> {
    // Prefer embedded binary JPEG avatar data.
    if let Some(bytes) = od_first_data_attr(record, od_attr_jpeg_photo()?)? {
        return Ok(bytes);
    }

    // Fallback to the existing macOS user picture path.
    if let Some(path) = od_first_string_attr(record, od_attr_picture()?)? {
        let path = path_or_file_url_to_pathbuf(&path)?;
        return Ok(std::fs::read(path)?);
    }

    Err(MacOsUserError::NoAvatar)
}

#[cfg(target_os = "macos")]
fn od_first_data_attr(
    record: &ODRecord,
    attr: &ODAttributeType,
) -> Result<Option<Vec<u8>>, MacOsUserError> {
    unsafe {
        let mut err: Option<Retained<NSError>> = None;

        let Some(values) = record.valuesForAttribute_error(Some(attr), Some(&mut err)) else {
            return if err.is_some() {
                Err(MacOsUserError::OpenDirectory(ns_error_string(err)))
            } else {
                Ok(None)
            };
        };

        for value in values.iter() {
            if let Some(data) = value.downcast_ref::<NSData>() {
                return Ok(Some(data_as_vec(data)));
            }
        }

        Ok(None)
    }
}

#[cfg(target_os = "macos")]
fn od_first_string_attr(
    record: &ODRecord,
    attr: &ODAttributeType,
) -> Result<Option<String>, MacOsUserError> {
    unsafe {
        let mut err: Option<Retained<NSError>> = None;

        let Some(values) = record.valuesForAttribute_error(Some(attr), Some(&mut err)) else {
            return if err.is_some() {
                Err(MacOsUserError::OpenDirectory(ns_error_string(err)))
            } else {
                Ok(None)
            };
        };

        for value in values.iter() {
            if let Some(s) = value.downcast_ref::<NSString>() {
                return Ok(Some(s.to_string()));
            }
        }

        Ok(None)
    }
}

#[cfg(target_os = "macos")]
fn data_as_vec(data: &NSData) -> Vec<u8> {
    // `NSData` values from OpenDirectory are immutable for the lifetime of `data`,
    // so copying the exposed byte slice into Rust-owned memory is safe here.
    data.to_vec()
}

#[cfg(target_os = "macos")]
fn encode_png(source: &[u8]) -> Result<Vec<u8>, MacOsUserError> {
    let image = ImageReader::new(Cursor::new(source))
        .with_guessed_format()?
        .decode()?;

    let mut png = Vec::new();
    image.write_to(&mut Cursor::new(&mut png), ImageFormat::Png)?;

    Ok(png)
}

#[cfg(target_os = "macos")]
fn path_or_file_url_to_pathbuf(s: &str) -> Result<PathBuf, MacOsUserError> {
    if let Ok(url) = url::Url::parse(s)
        && url.scheme() == "file"
    {
        return url
            .to_file_path()
            .map_err(|_| MacOsUserError::BadPicturePath(s.to_owned()));
    }

    Ok(PathBuf::from(s))
}

#[cfg(target_os = "macos")]
fn od_attr_jpeg_photo() -> Result<&'static ODAttributeType, MacOsUserError> {
    unsafe {
        kODAttributeTypeJPEGPhoto.ok_or_else(|| {
            MacOsUserError::OpenDirectory("kODAttributeTypeJPEGPhoto unavailable".to_owned())
        })
    }
}

#[cfg(target_os = "macos")]
fn od_attr_picture() -> Result<&'static ODAttributeType, MacOsUserError> {
    unsafe {
        kODAttributeTypePicture.ok_or_else(|| {
            MacOsUserError::OpenDirectory("kODAttributeTypePicture unavailable".to_owned())
        })
    }
}

#[cfg(target_os = "macos")]
fn od_attr_full_name() -> Result<&'static ODAttributeType, MacOsUserError> {
    unsafe {
        kODAttributeTypeFullName.ok_or_else(|| {
            MacOsUserError::OpenDirectory("kODAttributeTypeFullName unavailable".to_owned())
        })
    }
}

#[cfg(target_os = "macos")]
fn od_record_type_users() -> Result<&'static ODRecordType, MacOsUserError> {
    unsafe {
        kODRecordTypeUsers.ok_or_else(|| {
            MacOsUserError::OpenDirectory("kODRecordTypeUsers unavailable".to_owned())
        })
    }
}

#[cfg(target_os = "macos")]
fn ns_error_string(err: Option<Retained<NSError>>) -> String {
    err.map(|e| format!("{e:?}"))
        .unwrap_or_else(|| "unknown OpenDirectory error".to_owned())
}

#[derive(Debug, Error)]
enum MacOsUserError {
    #[cfg(not(target_os = "macos"))]
    #[error("macOS user details are only supported on macOS")]
    UnsupportedPlatform,

    #[cfg(target_os = "macos")]
    #[error("could not determine current user")]
    NoCurrentUser,

    #[cfg(target_os = "macos")]
    #[error("user has no JPEGPhoto or Picture avatar")]
    NoAvatar,

    #[cfg(target_os = "macos")]
    #[error("OpenDirectory error: {0}")]
    OpenDirectory(String),

    #[cfg(target_os = "macos")]
    #[error("bad Picture path: {0}")]
    BadPicturePath(String),

    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Image(#[from] image::ImageError),
}
