"""Hamilton scalar-first quaternions; active body-to-parent rotations, radians.
Single-attitude kernels; use jax.vmap for batches. Euler321 = [roll,pitch,yaw].
"""
import jax.numpy as jnp


def skew(vector):
    """skew(a) @ b == cross(a,b)."""
    x, y, z = vector
    zero = jnp.zeros_like(x)
    return jnp.stack((jnp.stack((zero,-z,y)),jnp.stack((z,zero,-x)),jnp.stack((-y,x,zero))))


def quaternion_multiply(a, b):
    """Hamilton product; R(a*b) = R(a) @ R(b) for unit quaternions."""
    aw, av, bw, bv = a[0], a[1:], b[0], b[1:]
    return jnp.concatenate(((aw*bw-jnp.dot(av,bv))[None],aw*bv+bw*av+jnp.cross(av,bv)))


def quaternion_conjugate(q):
    return jnp.concatenate((q[:1],-q[1:]))


def quaternion_inverse(q):
    norm2 = jnp.dot(q,q)
    return quaternion_conjugate(q)/jnp.where(norm2>0,norm2,jnp.nan)


def rotation_vector_to_quaternion(vector):
    """Axis times angle -> unit quaternion; smooth value/Jacobian at zero."""
    theta2 = jnp.dot(vector,vector)
    small = theta2 < 1e-6
    theta = jnp.sqrt(jnp.where(small,1.,theta2))
    scalar = jnp.where(small,1-theta2/8+theta2**2/384,jnp.cos(theta/2))
    scale = jnp.where(small,.5-theta2/48+theta2**2/3840,jnp.sin(theta/2)/theta)
    return normalize_quaternion(jnp.concatenate((scalar[None],scale*vector)))


def quaternion_error(reference, current):
    """Shortest error in parent axes: R(error) = R(ref) @ R(current).T.

    Sign selection is discontinuous at 180 degrees; q and -q give the same
    error rotation. This returns a quaternion, not a three-component residual.
    """
    error = normalize_quaternion(quaternion_multiply(normalize_quaternion(reference),
                                                     quaternion_conjugate(normalize_quaternion(current))))
    return jnp.where(error[0]<0,-error,error)


def rotate_vector(q, vector):
    return quaternion_to_matrix(q) @ vector


def euler321_to_matrix(euler):
    return quaternion_to_matrix(euler321_to_quaternion(euler))


def body_rates_to_euler321_rates(euler, omega, singularity_cos=1e-6):
    """Full Euler kinematics; NaN near pitch gimbal lock."""
    roll, pitch, _ = euler
    p, q, r = omega
    sr, cr, cp = jnp.sin(roll), jnp.cos(roll), jnp.cos(pitch)
    cp = jnp.where(jnp.abs(cp)>singularity_cos,cp,jnp.nan)
    mixed = sr*q+cr*r
    return jnp.stack((p+jnp.sin(pitch)/cp*mixed,cr*q-sr*r,mixed/cp))


def euler321_rates_to_body_rates(euler, rates):
    roll, pitch, _ = euler
    dr, dp, dy = rates
    sr, cr = jnp.sin(roll), jnp.cos(roll)
    return jnp.stack((dr-dy*jnp.sin(pitch), dp*cr+dy*sr*jnp.cos(pitch),
                      -dp*sr+dy*cr*jnp.cos(pitch)))


def normalize_quaternion(q):
    """No sign canonicalization: preserve continuity across w=0. Zero -> NaN."""
    norm = jnp.linalg.norm(q)
    return q / jnp.where(norm > 0, norm, jnp.nan)


def euler321_to_quaternion(euler):
    roll, pitch, yaw = euler / 2
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                      cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def quaternion_to_matrix(q):
    w, x, y, z = normalize_quaternion(q)
    return jnp.stack((jnp.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y))),
                      jnp.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x))),
                      jnp.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))))


def quaternion_to_euler321(q):
    """Principal Euler angles; at gimbal lock choose roll=0 (not differentiable)."""
    R = quaternion_to_matrix(q)
    cp = jnp.hypot(R[0, 0], R[1, 0])
    singular = cp < 1e-7
    # Safe inactive atan2 branches also avoid NaN derivatives at exact lock.
    roll = jnp.arctan2(jnp.where(singular, 0., R[2, 1]), jnp.where(singular, 1., R[2, 2]))
    yaw = jnp.arctan2(jnp.where(singular, -R[0, 1], R[1, 0]),
                      jnp.where(singular, R[1, 1], R[0, 0]))
    return jnp.stack((roll, jnp.arctan2(-R[2, 0], cp), yaw))
