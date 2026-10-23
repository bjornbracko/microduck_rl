"""Shared CPU BAM actuator setup for deployment rehearsal and the daemon twin."""
import mujoco
import numpy as np

# BAM M6 defaults — MUST mirror `_BAM_ACTUATOR_KWARGS` in
# src/mjlab_microduck/robot/microduck_constants.py (the actuator every policy is
# trained against in warp). Not imported from there: that module drags in
# mjlab/torch/warp (~16 s import) for a CPU rehearsal script. Locked by
# tests/test_infer_policy_bam.py.
BAM_MOTOR_NAME = "xl330"
BAM_MODEL = "m6"
BAM_KP_FW = 200.0                 # microduck's preserved firmware stiffness
BAM_VIN_RANGE = (6.5, 8.2)        # per-env battery voltage DR in training
BAM_VIN_DROP_GAIN_RANGE = (0.0, 0.2)  # load-dependent sag V_drop = gain * sum|tau|
BAM_VIN_MIN = 6.0                 # floor on effective voltage after sag
BAM_MAX_CURRENT = None            # training runs WITHOUT the firmware current limiter
# Stiff joint-friction constraint, copied from bam.mjlab.BamActuator
# (stiff_frictionloss=True in training): warp has no noslip solver, so BAM
# stiffens frictionloss so a statically-held joint does not creep. Mirrored
# here so CPU and warp apply the same friction budget the same way.
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_bam_model(kp_fw: float, vin: float, max_current):
    """Build the BAM M6 model + XL330 voltage-controlled actuator."""
    from bam.model import load_model
    bam_model = load_model(motor_name=BAM_MOTOR_NAME, model=BAM_MODEL)
    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin
    bam_model.actuator.max_current = max_current if (max_current and max_current > 0) else None
    return bam_model


def load_mujoco_with_bam(xml_path: str, bam_model, timestep: float, vin_drop_gain, vin_min):
    """Load the scene and hand every non-passive actuator to bam.mujoco.MujocoController.

    Mirrors bam.mjlab.BamActuator.edit_spec (what warp does at training time):
    position actuators -> torque motors with the voltage-bounded forcerange,
    joint damping/frictionloss zeroed (BAM rewrites them every step), stiff
    friction constraint. Armature is set on the dofs by MujocoController.
    Returns (model, data, bam_ctrl, actuator_names).
    """
    from bam.mujoco import MujocoController

    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(xml_path)
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))  # MjsJoint expects a (3,1) array
                joint.frictionloss = 0.0
                joint.solref_friction = BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
                break

    model = spec.compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    bam_ctrl = MujocoController(bam_model, names, model, data,
                                vin_drop_gain=vin_drop_gain, vin_min=vin_min)
    print(f"BAM {BAM_MODEL} actuators on {len(names)} joints: kt={kt:.4f} R={R:.4f} "
          f"vin={bam_model.actuator.vin:.2f}V kp_fw={bam_model.actuator.kp:.0f} "
          f"vin_drop_gain={vin_drop_gain} vin_min={vin_min} "
          f"max_current={bam_model.actuator.max_current} forcerange=+/-{force_limit:.3f}Nm "
          f"armature={bam_model.actuator.get_extra_inertia():.2e}")
    return model, data, bam_ctrl, names


