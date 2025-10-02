pub(crate) mod buffers;
pub(crate) mod compute;

pub fn aabb_visible(view_proj: glam::Mat4, min: glam::Vec3, max: glam::Vec3) -> bool {
    // Precompute the 8 corners in clip space
    let corners: [glam::Vec4; 8] = [
        view_proj * min.extend(1.0),
        view_proj * glam::vec3(max.x, min.y, min.z).extend(1.0),
        view_proj * glam::vec3(min.x, max.y, min.z).extend(1.0),
        view_proj * glam::vec3(max.x, max.y, min.z).extend(1.0),
        view_proj * glam::vec3(min.x, min.y, max.z).extend(1.0),
        view_proj * glam::vec3(max.x, min.y, max.z).extend(1.0),
        view_proj * glam::vec3(min.x, max.y, max.z).extend(1.0),
        view_proj * max.extend(1.0),
    ];

    // Check each frustum plane
    for plane in 0..6 {
        let mut all_outside = true;

        for &clip in &corners {
            if clip.w <= 0.0 {
                continue;
            }

            let inside = match plane {
                0 => clip.x >= -clip.w, // left
                1 => clip.x <= clip.w,  // right
                2 => clip.y >= -clip.w, // bottom
                3 => clip.y <= clip.w,  // top
                4 => clip.z <= clip.w,  // near
                5 => clip.z >= 0.0,     // far
                _ => unreachable!(),
            };

            if inside {
                all_outside = false;
                break;
            }
        }

        if all_outside {
            return false;
        }
    }

    true
}
