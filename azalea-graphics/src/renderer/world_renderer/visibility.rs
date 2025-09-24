pub fn aabb_visible(view_proj: glam::Mat4, min: glam::Vec3, max: glam::Vec3) -> bool {
    let corners = [
        glam::vec3(min.x, min.y, min.z),
        glam::vec3(max.x, min.y, min.z),
        glam::vec3(min.x, max.y, min.z),
        glam::vec3(max.x, max.y, min.z),
        glam::vec3(min.x, min.y, max.z),
        glam::vec3(max.x, min.y, max.z),
        glam::vec3(min.x, max.y, max.z),
        glam::vec3(max.x, max.y, max.z),
    ];

    let mut all_outside = true;

    for c in corners {
        let clip = view_proj * c.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;

        if ndc.x >= -1.0
            || ndc.x <= 1.0
            || ndc.y >= -1.0
            || ndc.y <= 1.0
            || ndc.z >= 0.0
            || ndc.z <= 1.0
        {
            all_outside = false;
        }
    }

    !all_outside
}


