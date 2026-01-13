/// Buffer
pub mod buffer;
/// Load audio file.
pub mod decode;
/// Load audio file.
pub mod error;

pub use buffer::Buffer;
pub use decode::load;
pub use error::AudioError;
