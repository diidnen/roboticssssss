"""Common random numbers under possibly different native termination times."""


def matched_native_prefixes(schedules):
    if not schedules or any(not x for x in schedules):
        return False
    canonical = max(schedules,key=len)
    for schedule in schedules:
        if [r['step'] for r in schedule] != list(range(1,1+10*len(schedule),10)):
            return False
        if schedule != canonical[:len(schedule)]:
            return False
    return True
