#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Left = 0,
    Right = 1,
}

impl Arm {
    pub fn opposite(self) -> Arm {
        if self == Self::Left {
            Self::Right
        } else {
            Self::Left
        }
    }

    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn to_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}
