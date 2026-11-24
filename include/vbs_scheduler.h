#ifndef VBS_SCHEDULER_H
#define VBS_SCHEDULER_H

/* Learning-rate schedule, written by the engine's own self-edit loop.
 * The rate decays in steps as training goes on; see vbs_scheduler.c
 * for the exact schedule. */
float vbs_scheduler_lr(float base_lr, int epoch);

#endif /* VBS_SCHEDULER_H */
