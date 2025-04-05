import pytest
import jax.numpy as jnp
import jaxlie

# Import directly from the coll package level
from pyroki.coll import collide, Sphere, HalfSpace, Capsule, Box


# Fixtures or helper functions could be added later for more complex setups


def test_sphere_sphere_touching():
    """Test two spheres touching at the origin."""
    sphere1 = Sphere.from_center_and_radius(
        center=jnp.array([-0.5, 0, 0]), radius=jnp.array(0.5)
    )
    sphere2 = Sphere.from_center_and_radius(
        center=jnp.array([0.5, 0, 0]), radius=jnp.array(0.5)
    )
    expected_dist = 0.0
    dist = collide(sphere1, sphere2)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_sphere_sphere_separated():
    """Test two spheres separated along the x-axis."""
    sphere1 = Sphere.from_center_and_radius(
        center=jnp.array([-1.0, 0, 0]), radius=jnp.array(0.5)
    )
    sphere2 = Sphere.from_center_and_radius(
        center=jnp.array([1.0, 0, 0]), radius=jnp.array(0.5)
    )
    # Distance between centers is 2.0. Sum of radii is 1.0. Expected dist = 2.0 - 1.0 = 1.0
    expected_dist = 1.0
    dist = collide(sphere1, sphere2)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_sphere_sphere_overlapping():
    """Test two spheres overlapping."""
    sphere1 = Sphere.from_center_and_radius(
        center=jnp.array([-0.2, 0, 0]), radius=jnp.array(0.5)
    )
    sphere2 = Sphere.from_center_and_radius(
        center=jnp.array([0.2, 0, 0]), radius=jnp.array(0.5)
    )
    # Distance between centers is 0.4. Sum of radii is 1.0. Expected dist = 0.4 - 1.0 = -0.6
    expected_dist = -0.6
    dist = collide(sphere1, sphere2)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_halfspace_sphere_touching():
    """Test a sphere touching a halfspace defined by the XY plane (normal +Z)."""
    # Halfspace at z=0, normal pointing +z
    halfspace = HalfSpace.from_point_and_normal(
        point=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0])
    )
    # Sphere radius 0.5, center at z=0.5
    sphere = Sphere.from_center_and_radius(
        center=jnp.array([0, 0, 0.5]), radius=jnp.array(0.5)
    )
    expected_dist = 0.0
    dist = collide(halfspace, sphere)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_halfspace_sphere_separated():
    """Test a sphere separated from a halfspace defined by the XY plane (normal +Z)."""
    # Halfspace at z=0, normal pointing +z
    halfspace = HalfSpace.from_point_and_normal(
        point=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0])
    )
    # Sphere radius 0.5, center at z=1.0
    sphere = Sphere.from_center_and_radius(
        center=jnp.array([0, 0, 1.0]), radius=jnp.array(0.5)
    )
    # Distance from center to plane is 1.0. Expected dist = 1.0 - 0.5 = 0.5
    expected_dist = 0.5
    dist = collide(halfspace, sphere)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_halfspace_sphere_penetrating():
    """Test a sphere penetrating a halfspace defined by the XY plane (normal +Z)."""
    # Halfspace at z=0, normal pointing +z
    halfspace = HalfSpace.from_point_and_normal(
        point=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0])
    )
    # Sphere radius 0.5, center at z=0.2
    sphere = Sphere.from_center_and_radius(
        center=jnp.array([0, 0, 0.2]), radius=jnp.array(0.5)
    )
    # Distance from center to plane is 0.2. Expected dist = 0.2 - 0.5 = -0.3
    expected_dist = -0.3
    dist = collide(halfspace, sphere)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_capsule_capsule_touching_parallel():
    """Test two parallel capsules touching end-to-end."""
    # Capsules along X-axis, radius 0.1, half-length 0.5
    pose1 = jaxlie.SE3.from_translation(
        jnp.array([-0.1, 0, 0])
    )  # Back to original pose
    pose2 = jaxlie.SE3.from_translation(jnp.array([0.1, 0, 0]))  # Back to original pose
    cap1 = Capsule.from_center_radius_height(
        center=pose1.translation(),
        orientation_mat=pose1.rotation().as_matrix(),
        radius=jnp.array(0.2),
        height=jnp.array(1.0),  # Back to original height
    )
    cap2 = Capsule.from_center_radius_height(
        center=pose2.translation(),
        orientation_mat=pose2.rotation().as_matrix(),
        radius=jnp.array(0.2),
        height=jnp.array(1.0),  # Back to original height
    )
    expected_dist = -0.2  # Correct expectation
    dist = collide(cap1, cap2)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_capsule_capsule_separated_parallel():
    """Test two parallel capsules separated."""
    pose1 = jaxlie.SE3.from_translation(
        jnp.array([-0.5, 0, 0])
    )  # Back to original pose
    pose2 = jaxlie.SE3.from_translation(jnp.array([0.5, 0, 0]))  # Back to original pose
    cap1 = Capsule.from_center_radius_height(
        center=pose1.translation(),
        orientation_mat=pose1.rotation().as_matrix(),
        radius=jnp.array(0.2),
        height=jnp.array(1.0),  # Back to original height
    )
    cap2 = Capsule.from_center_radius_height(
        center=pose2.translation(),
        orientation_mat=pose2.rotation().as_matrix(),
        radius=jnp.array(0.2),
        height=jnp.array(1.0),  # Back to original height
    )
    expected_dist = 0.6  # Correct expectation
    dist = collide(cap1, cap2)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_halfspace_capsule_touching():
    """Test capsule end touching a halfspace (XY plane, normal +Z)."""
    halfspace = HalfSpace.from_point_and_normal(
        point=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0])
    )
    # Capsule along Z-axis, radius 0.1, half-length 0.5, center at z=0.6
    # Ends at z=0.1 and z=1.1. Lower end sphere center at z=0.1
    # Use identity rotation to align capsule local Z with world Z
    pose = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.identity(),  # CHANGED rotation
        translation=jnp.array([0, 0, 0.6]),
    )
    cap = Capsule.from_center_radius_height(
        center=pose.translation(),
        orientation_mat=pose.rotation().as_matrix(),
        radius=jnp.array(0.1),
        height=jnp.array(0.5 * 2.0),
    )
    expected_dist = 0.0  # Lower sphere center dist 0.1 - radius 0.1
    dist = collide(halfspace, cap)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_halfspace_capsule_rotated():
    """Test capsule rotated relative to halfspace."""
    halfspace = HalfSpace.from_point_and_normal(
        point=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0])
    )
    # Capsule radius 0.1, half-length 0.5, center at z=0.5
    # Rotated 45 deg around Y axis. Lowest point is lower end sphere bottom: z = 0.1464... - 0.1 = 0.0464...
    pose = jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.from_y_radians(jnp.pi / 4),
        translation=jnp.array([0, 0, 0.5]),
    )
    cap = Capsule.from_center_radius_height(
        center=pose.translation(),
        orientation_mat=pose.rotation().as_matrix(),
        radius=jnp.array(0.1),
        height=jnp.array(0.5 * 2.0),
    )
    # expected_dist = 0.05 # Old inaccurate expectation
    expected_dist_precise = 0.1464466094 - 0.1  # z_end2 - radius
    dist = collide(halfspace, cap)
    assert dist is not None
    # Use pytest.approx for cleaner floating point comparison
    assert dist == pytest.approx(expected_dist_precise, abs=1e-6)
    # assert jnp.allclose(dist, expected_dist, atol=1e-6) # Old assertion


def test_sphere_capsule_touching():
    """Test sphere touching capsule side."""
    sphere_pose = jaxlie.SE3.from_translation(jnp.array([0.0, 0.3, 0]))
    sphere = Sphere.from_center_and_radius(
        center=sphere_pose.translation(), radius=jnp.array(0.2)
    )
    # Capsule along X-axis, radius 0.1, half-length 0.5
    cap_pose = jaxlie.SE3.identity()  # Axis from (-0.5,0,0) to (0.5,0,0)
    cap = Capsule.from_center_radius_height(
        center=cap_pose.translation(),
        orientation_mat=cap_pose.rotation().as_matrix(),
        radius=jnp.array(0.1),
        height=jnp.array(0.5 * 2.0),
    )
    # Closest point on axis is (0,0,0). Dist sphere center to axis is 0.3.
    expected_dist = 0.0  # Center dist 0.3 - radii sum (0.2 + 0.1)
    dist = collide(sphere, cap)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


# --- Box Tests ---


def test_halfspace_box_touching():
    """Test box face touching a halfspace (XY plane, normal +Z)."""
    halfspace = HalfSpace.from_point_and_normal(
        point=jnp.zeros(3), normal=jnp.array([0.0, 0.0, 1.0])
    )
    # Box extents (1,1,1), so half-extents 0.5. Center at z=0.5
    box_pose = jaxlie.SE3.from_translation(jnp.array([0, 0, 0.5]))
    box = Box.from_center_extents_pose(
        pose=box_pose, extents=jnp.array([1.0, 1.0, 1.0])
    )
    # Lowest vertex z = 0.5 - 0.5 = 0.0
    expected_dist = 0.0
    dist = collide(halfspace, box)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


def test_sphere_box_touching_face():
    """Test sphere touching center of box face."""
    # Box extents (1,1,1), center origin. Face at x=0.5
    box_pose = jaxlie.SE3.identity()
    box = Box.from_center_extents_pose(
        pose=box_pose, extents=jnp.array([1.0, 1.0, 1.0])
    )
    # Sphere radius 0.2, center at x=0.7
    sphere_pose = jaxlie.SE3.from_translation(jnp.array([0.7, 0, 0]))
    sphere = Sphere.from_center_and_radius(
        center=sphere_pose.translation(), radius=jnp.array(0.2)
    )
    # Closest point on box is (0.5, 0, 0). Dist sphere center to point = 0.7 - 0.5 = 0.2
    expected_dist = 0.0  # Dist 0.2 - radius 0.2
    dist = collide(sphere, box)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


# Note: Capsule-Box distance is approximate in current implementation
def test_capsule_box_touching_face_approx():
    """Test capsule end touching center of box face (approx distance)."""
    # Box extents (1,1,1), center origin. Face at x=0.5
    box_pose = jaxlie.SE3.identity()
    box = Box.from_center_extents_pose(
        pose=box_pose, extents=jnp.array([1.0, 1.0, 1.0])
    )
    # Capsule along X-axis, radius 0.1, half-length 0.5. Center at x=1.1
    # Ends at x=0.6 and x=1.6. Front sphere center at x=0.6
    cap_center = jnp.array([1.1, 0, 0])
    # Rotation to align local Z axis (capsule default) with world X axis
    # Rotate around Y by -90 degrees
    orientation_matrix = jaxlie.SO3.from_y_radians(-jnp.pi / 2).as_matrix()
    cap = Capsule.from_center_radius_height(
        center=cap_center,
        orientation_mat=orientation_matrix,  # Use correct orientation
        radius=jnp.array(0.1),
        height=jnp.array(0.5 * 2.0),
    )
    # Approx calculation based on implementation:
    # Closest point on box to cap center (1.1,0,0) is (0.5,0,0).
    # Capsule axis is now X. Segment is [0.6, 1.6] along X.
    # Closest point on cap axis segment to (0.5,0,0) is (0.6,0,0).
    # Dist between (0.5,0,0) and (0.6,0,0) is 0.1.
    expected_dist = 0.0  # Dist 0.1 - radius 0.1
    dist = collide(cap, box)
    assert dist is not None
    assert jnp.allclose(dist, expected_dist, atol=1e-6)


# --- Batching Test ---


def test_batch_sphere_sphere():
    """Test collision with a batch of sphere pairs."""
    # Batch 1: Touching
    center1a = jnp.array([-0.5, 0, 0])
    center2a = jnp.array([0.5, 0, 0])
    # Batch 2: Separated
    center1b = jnp.array([-1.0, 1, 0])
    center2b = jnp.array([1.0, 1, 0])

    # Stack centers for batching
    centers1 = jnp.stack([center1a, center1b])
    centers2 = jnp.stack([center2a, center2b])

    # Create batched spheres (radius broadcast)
    spheres1 = Sphere.from_center_and_radius(center=centers1, radius=jnp.array(0.5))
    spheres2 = Sphere.from_center_and_radius(center=centers2, radius=jnp.array(0.5))

    expected_dists = jnp.array(
        [0.0, 1.0]
    )  # Batch 1 touching (0.0), Batch 2 separated (1.0)
    dists = collide(spheres1, spheres2)

    assert dists is not None
    assert dists.shape == (2,)
    assert jnp.allclose(dists, expected_dists, atol=1e-6)
