# Encodings
RAW_ENCODING =                  0
COPY_RECTANGLE_ENCODING =       1
RRE_ENCODING =                  2
CORRE_ENCODING =                4
HEXTILE_ENCODING =              5
ZLIB_ENCODING =                 6
TIGHT_ENCODING =                7
ZLIBHEX_ENCODING =              8
ZRLE_ENCODING =                 16
#0xffffff00 to 0xffffffff tight options
PSEUDO_CURSOR_ENCODING =        -239

# Keycodes
KEY_BackSpace = 0xff08
KEY_Tab =       0xff09
KEY_Return =    0xff0d
KEY_Escape =    0xff1b
KEY_Insert =    0xff63
KEY_Delete =    0xffff
KEY_Home =      0xff50
KEY_End =       0xff57
KEY_PageUp =    0xff55
KEY_PageDown =  0xff56
KEY_Left =      0xff51
KEY_Up =        0xff52
KEY_Right =     0xff53
KEY_Down =      0xff54
KEY_F1 =        0xffbe
KEY_F2 =        0xffbf
KEY_F3 =        0xffc0
KEY_F4 =        0xffc1
KEY_F5 =        0xffc2
KEY_F6 =        0xffc3
KEY_F7 =        0xffc4
KEY_F8 =        0xffc5
KEY_F9 =        0xffc6
KEY_F10 =       0xffc7
KEY_F11 =       0xffc8
KEY_F12 =       0xffc9
KEY_F13 =       0xFFCA
KEY_F14 =       0xFFCB
KEY_F15 =       0xFFCC
KEY_F16 =       0xFFCD
KEY_F17 =       0xFFCE
KEY_F18 =       0xFFCF
KEY_F19 =       0xFFD0
KEY_F20 =       0xFFD1
KEY_ShiftLeft = 0xffe1
KEY_ShiftRight = 0xffe2
KEY_ControlLeft = 0xffe3
KEY_ControlRight = 0xffe4
KEY_MetaLeft =  0xffe7
KEY_MetaRight = 0xffe8
KEY_AltLeft =   0xffe9
KEY_AltRight =  0xffea

KEY_Scroll_Lock = 0xFF14
KEY_Sys_Req =   0xFF15
KEY_Num_Lock =  0xFF7F
KEY_Caps_Lock = 0xFFE5
KEY_Pause =     0xFF13
KEY_Super_L =   0xFFEB
KEY_Super_R =   0xFFEC
KEY_Hyper_L =   0xFFED
KEY_Hyper_R =   0xFFEE

KEY_KP_0 =      0xFFB0
KEY_KP_1 =      0xFFB1
KEY_KP_2 =      0xFFB2
KEY_KP_3 =      0xFFB3
KEY_KP_4 =      0xFFB4
KEY_KP_5 =      0xFFB5
KEY_KP_6 =      0xFFB6
KEY_KP_7 =      0xFFB7
KEY_KP_8 =      0xFFB8
KEY_KP_9 =      0xFFB9
KEY_KP_Enter =  0xFF8D

KEY_ForwardSlash = 0x002F
KEY_BackSlash = 0x005C
KEY_SpaceBar=   0x0020

# TODO: build this programmatically?
KEYMAP = {
    'bsp': KEY_BackSpace,
    'tab': KEY_Tab,
    'return': KEY_Return,
    'enter': KEY_Return,
    'esc': KEY_Escape,
    'ins': KEY_Insert,
    'delete': KEY_Delete,
    'del': KEY_Delete,
    'home': KEY_Home,
    'end': KEY_End,
    'pgup': KEY_PageUp,
    'pgdn': KEY_PageDown,
    'ArrowLeft': KEY_Left,
    'left': KEY_Left,
    'ArrowUp': KEY_Up,
    'up': KEY_Up,
    'ArrowRight': KEY_Right,
    'right': KEY_Right,
    'ArrowDown': KEY_Down,
    'down': KEY_Down,

    'slash': KEY_BackSlash,
    'bslash': KEY_BackSlash,
    'fslash': KEY_ForwardSlash,
    'spacebar': KEY_SpaceBar,
    'space': KEY_SpaceBar,
    'sb': KEY_SpaceBar,

    'f1': KEY_F1,
    'f2': KEY_F2,
    'f3': KEY_F3,
    'f4': KEY_F4,
    'f5': KEY_F5,
    'f6': KEY_F6,
    'f7': KEY_F7,
    'f8': KEY_F8,
    'f9': KEY_F9,
    'f10': KEY_F10,
    'f11': KEY_F11,
    'f12': KEY_F12,
    'f13': KEY_F13,
    'f14': KEY_F14,
    'f15': KEY_F15,
    'f16': KEY_F16,
    'f17': KEY_F17,
    'f18': KEY_F18,
    'f19': KEY_F19,
    'f20': KEY_F20,

    'lshift': KEY_ShiftLeft,
    'shift': KEY_ShiftLeft,
    'rshift': KEY_ShiftRight,
    'lctrl': KEY_ControlLeft,
    'ctrl': KEY_ControlLeft,
    'rctrl': KEY_ControlRight,
    'lmeta': KEY_MetaLeft,
    'meta': KEY_MetaLeft,
    'rmeta': KEY_MetaRight,
    'lalt': KEY_AltLeft,
    'alt': KEY_AltLeft,
    'ralt': KEY_AltRight,
    'scrlk': KEY_Scroll_Lock,
    'sysrq': KEY_Sys_Req,
    'numlk': KEY_Num_Lock,
    'caplk': KEY_Caps_Lock,
    'pause': KEY_Pause,
    'lsuper': KEY_Super_L,
    'super': KEY_Super_L,
    'rsuper': KEY_Super_R,
    'lhyper': KEY_Hyper_L,
    'hyper': KEY_Hyper_L,
    'rhyper': KEY_Hyper_R,

    'kp0': KEY_KP_0,
    'kp1': KEY_KP_1,
    'kp2': KEY_KP_2,
    'kp3': KEY_KP_3,
    'kp4': KEY_KP_4,
    'kp5': KEY_KP_5,
    'kp6': KEY_KP_6,
    'kp7': KEY_KP_7,
    'kp8': KEY_KP_8,
    'kp9': KEY_KP_9,
    'kpenter': KEY_KP_Enter,
}
