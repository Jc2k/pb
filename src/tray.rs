//! macOS menu bar status item for a `pb serve` instance.

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct TrayArgs {
    pub host: String,
    pub port: u16,
}

pub fn run(args: TrayArgs) -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = args;
        anyhow::bail!("pb tray is only supported on macOS");
    }

    #[cfg(target_os = "macos")]
    {
        macos::run(args)
    }
}

#[cfg(target_os = "macos")]
mod macos {
    #![allow(unsafe_op_in_unsafe_fn)]
    use super::TrayArgs;
    use anyhow::Result;
    use std::ffi::{CString, c_char, c_double, c_void};
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::OnceLock;
    use std::time::Duration;

    type Id = *mut c_void;
    type Sel = *mut c_void;
    type Bool = i8;

    const ICON_BYTES: &[u8] = include_bytes!("../public/icon-192.png");
    const TRAY_ICON_SIZE: c_double = 18.0;

    static WEB_URL: OnceLock<String> = OnceLock::new();
    static STATUS_HOST: OnceLock<String> = OnceLock::new();
    static STATUS_PORT: OnceLock<u16> = OnceLock::new();
    static STATUS_BUTTON: OnceLock<usize> = OnceLock::new();
    static STATUS_ITEM: OnceLock<usize> = OnceLock::new();
    static STATUS_DELEGATE: OnceLock<usize> = OnceLock::new();

    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn objc_allocateClassPair(superclass: Id, name: *const c_char, extra_bytes: usize) -> Id;
        fn objc_registerClassPair(cls: Id);
        fn class_addMethod(cls: Id, name: Sel, imp: *const c_void, types: *const c_char) -> Bool;
        fn sel_registerName(name: *const c_char) -> Sel;
        // objc_msgSend must be cast to the exact Objective-C method signature for
        // each call site. Calling it through a single C-variadic declaration is
        // undefined on macOS/arm64, especially for floating-point arguments.
        fn objc_msgSend();
    }

    pub fn run(args: TrayArgs) -> Result<()> {
        let web_url = format!("http://{}:{}/", args.host, args.port);
        WEB_URL
            .set(web_url)
            .map_err(|_| anyhow::anyhow!("tray URL already initialized"))?;
        STATUS_HOST
            .set(args.host)
            .map_err(|_| anyhow::anyhow!("tray host already initialized"))?;
        STATUS_PORT
            .set(args.port)
            .map_err(|_| anyhow::anyhow!("tray port already initialized"))?;

        unsafe {
            let app: Id = msg_send_id0(class("NSApplication"), sel("sharedApplication"));
            msg_send_void1_i64(app, sel("setActivationPolicy:"), 1_i64);

            let delegate_class = create_delegate_class()?;
            let delegate: Id = msg_send_id0(delegate_class, sel("new"));
            STATUS_DELEGATE
                .set(delegate as usize)
                .map_err(|_| anyhow::anyhow!("tray delegate already initialized"))?;

            let status_bar: Id = msg_send_id0(class("NSStatusBar"), sel("systemStatusBar"));
            let status_item: Id = msg_send_id1_f64(
                status_bar,
                sel("statusItemWithLength:"),
                -1.0_f64 as c_double,
            );
            msg_send_id0(status_item, sel("retain"));
            STATUS_ITEM
                .set(status_item as usize)
                .map_err(|_| anyhow::anyhow!("tray status item already initialized"))?;
            let button: Id = msg_send_id0(status_item, sel("button"));
            STATUS_BUTTON
                .set(button as usize)
                .map_err(|_| anyhow::anyhow!("tray button already initialized"))?;
            msg_send_void1_id(button, sel("setImage:"), status_icon());
            set_button_title("");
            msg_send_void1_id(button, sel("setToolTip:"), ns_string("pb idle"));
            msg_send_void1_id(button, sel("setTarget:"), delegate);
            msg_send_void1_sel(button, sel("setAction:"), sel("pbStatusItemClicked:"));

            let _timer = msg_send_id5_timer(
                class("NSTimer"),
                sel("scheduledTimerWithTimeInterval:target:selector:userInfo:repeats:"),
                2.0_f64 as c_double,
                delegate,
                sel("pbStatusTimerFired:"),
                std::ptr::null_mut::<c_void>(),
                1_i8,
            );

            update_status_item();
            msg_send_void0(app, sel("run"));
        }

        Ok(())
    }

    unsafe fn create_delegate_class() -> Result<Id> {
        let superclass = class("NSObject");
        let name = CString::new("PbTrayDelegate").unwrap();
        let cls = unsafe { objc_allocateClassPair(superclass, name.as_ptr(), 0) };
        if cls.is_null() {
            return Ok(class("PbTrayDelegate"));
        }

        let clicked_types = CString::new("v@:@").unwrap();
        let tick_types = CString::new("v@:@").unwrap();
        unsafe {
            class_addMethod(
                cls,
                sel("pbStatusItemClicked:"),
                clicked as *const c_void,
                clicked_types.as_ptr(),
            );
            class_addMethod(
                cls,
                sel("pbStatusTimerFired:"),
                timer_fired as *const c_void,
                tick_types.as_ptr(),
            );
            objc_registerClassPair(cls);
        }
        Ok(cls)
    }

    extern "C" fn clicked(_this: Id, _cmd: Sel, _sender: Id) {
        unsafe {
            let Some(url) = WEB_URL.get() else {
                return;
            };
            let workspace: Id = msg_send_id0(class("NSWorkspace"), sel("sharedWorkspace"));
            let ns_url: Id = msg_send_id1_id(class("NSURL"), sel("URLWithString:"), ns_string(url));
            msg_send_void1_id(workspace, sel("openURL:"), ns_url);
        }
    }

    extern "C" fn timer_fired(_this: Id, _cmd: Sel, _timer: Id) {
        update_status_item();
    }

    fn update_status_item() {
        let busy = server_busy();
        unsafe {
            if busy {
                set_button_title("•");
                if let Some(button) = STATUS_BUTTON.get() {
                    msg_send_void1_id(*button as Id, sel("setToolTip:"), ns_string("pb busy"));
                }
            } else {
                set_button_title("");
                if let Some(button) = STATUS_BUTTON.get() {
                    msg_send_void1_id(*button as Id, sel("setToolTip:"), ns_string("pb idle"));
                }
            }
        }
    }

    unsafe fn set_button_title(title: &str) {
        if let Some(button) = STATUS_BUTTON.get() {
            msg_send_void1_id(*button as Id, sel("setTitle:"), ns_string(title));
        }
    }

    fn server_busy() -> bool {
        let Some(host) = STATUS_HOST.get() else {
            return false;
        };
        let Some(port) = STATUS_PORT.get() else {
            return false;
        };
        let Ok(mut stream) = TcpStream::connect((host.as_str(), *port)) else {
            return false;
        };
        let _ = stream.set_read_timeout(Some(Duration::from_millis(750)));
        let _ = stream.set_write_timeout(Some(Duration::from_millis(750)));
        if write!(
            stream,
            "GET /api/status HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
        )
        .is_err()
        {
            return false;
        }
        let mut response = String::new();
        if stream.read_to_string(&mut response).is_err() {
            return false;
        }
        response.contains("\"busy\":true")
    }

    unsafe fn class(name: &str) -> Id {
        let name = CString::new(name).unwrap();
        unsafe { objc_getClass(name.as_ptr()) }
    }

    unsafe fn sel(name: &str) -> Sel {
        let name = CString::new(name).unwrap();
        unsafe { sel_registerName(name.as_ptr()) }
    }

    unsafe fn ns_string(value: &str) -> Id {
        let ns_string: Id = msg_send_id0(class("NSString"), sel("alloc"));
        let c_string = CString::new(value).unwrap();
        msg_send_id3_ptr_usize_u64(
            ns_string,
            sel("initWithBytes:length:encoding:"),
            c_string.as_ptr().cast::<c_void>(),
            value.len(),
            4_u64,
        )
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NsSize {
        width: c_double,
        height: c_double,
    }

    unsafe fn status_icon() -> Id {
        let data = msg_send_id2_ptr_usize(
            class("NSData"),
            sel("dataWithBytes:length:"),
            ICON_BYTES.as_ptr().cast::<c_void>(),
            ICON_BYTES.len(),
        );
        let image_alloc: Id = msg_send_id0(class("NSImage"), sel("alloc"));
        let image: Id = msg_send_id1_id(image_alloc, sel("initWithData:"), data);
        msg_send_void1_size(
            image,
            sel("setSize:"),
            NsSize {
                width: TRAY_ICON_SIZE,
                height: TRAY_ICON_SIZE,
            },
        );
        msg_send_void1_bool(image, sel("setTemplate:"), 1_i8);
        image
    }

    unsafe fn msg_send_id0(receiver: Id, selector: Sel) -> Id {
        type FnPtr = unsafe extern "C" fn(Id, Sel) -> Id;
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector) }
    }

    unsafe fn msg_send_void0(receiver: Id, selector: Sel) {
        type FnPtr = unsafe extern "C" fn(Id, Sel);
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector) }
    }

    unsafe fn msg_send_id1_id(receiver: Id, selector: Sel, arg: Id) -> Id {
        type FnPtr = unsafe extern "C" fn(Id, Sel, Id) -> Id;
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }

    unsafe fn msg_send_id1_f64(receiver: Id, selector: Sel, arg: c_double) -> Id {
        type FnPtr = unsafe extern "C" fn(Id, Sel, c_double) -> Id;
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }

    unsafe fn msg_send_id2_ptr_usize(
        receiver: Id,
        selector: Sel,
        ptr: *const c_void,
        len: usize,
    ) -> Id {
        type FnPtr = unsafe extern "C" fn(Id, Sel, *const c_void, usize) -> Id;
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, ptr, len) }
    }

    unsafe fn msg_send_id3_ptr_usize_u64(
        receiver: Id,
        selector: Sel,
        ptr: *const c_void,
        len: usize,
        encoding: u64,
    ) -> Id {
        type FnPtr = unsafe extern "C" fn(Id, Sel, *const c_void, usize, u64) -> Id;
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, ptr, len, encoding) }
    }

    unsafe fn msg_send_id5_timer(
        receiver: Id,
        selector: Sel,
        interval: c_double,
        target: Id,
        timer_selector: Sel,
        user_info: Id,
        repeats: Bool,
    ) -> Id {
        type FnPtr = unsafe extern "C" fn(Id, Sel, c_double, Id, Sel, Id, Bool) -> Id;
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe {
            f(
                receiver,
                selector,
                interval,
                target,
                timer_selector,
                user_info,
                repeats,
            )
        }
    }

    unsafe fn msg_send_void1_id(receiver: Id, selector: Sel, arg: Id) {
        type FnPtr = unsafe extern "C" fn(Id, Sel, Id);
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }

    unsafe fn msg_send_void1_sel(receiver: Id, selector: Sel, arg: Sel) {
        type FnPtr = unsafe extern "C" fn(Id, Sel, Sel);
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }

    unsafe fn msg_send_void1_i64(receiver: Id, selector: Sel, arg: i64) {
        type FnPtr = unsafe extern "C" fn(Id, Sel, i64);
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }

    unsafe fn msg_send_void1_bool(receiver: Id, selector: Sel, arg: Bool) {
        type FnPtr = unsafe extern "C" fn(Id, Sel, Bool);
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }

    unsafe fn msg_send_void1_size(receiver: Id, selector: Sel, arg: NsSize) {
        type FnPtr = unsafe extern "C" fn(Id, Sel, NsSize);
        let f: FnPtr = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
        unsafe { f(receiver, selector, arg) }
    }
}
