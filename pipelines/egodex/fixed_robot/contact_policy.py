"""Sustained unintended contact between fixed robot and scene bodies."""


def declared_surface_pairs(model, hand, obj, binding):
    surface = binding.get("contact_geom")
    if surface is None:
        return set()
    if binding["mode"] != "contact":
        raise ValueError("contact_geom requires a contact phase")
    target = model.geom(surface).id

    def belongs(body, root):
        while body and body != root:
            body = model.body_parentid[body]
        return body == root

    if not belongs(model.geom_bodyid[target], obj):
        raise ValueError(f"{surface} must belong to {binding['object']}")
    return {tuple(sorted((i, target))) for i, body in enumerate(model.geom_bodyid) if belongs(body, hand)}


class FixedContactMonitor:
    # Sum force across the contact manifold before testing its duration.
    minimum_force_n = 1.0
    minimum_seconds = 0.05

    def __init__(self, timestep):
        self.timestep = timestep
        self.active = {}
        self.reported = set()
        self.failures = []

    def step(self, timestamp, forces):
        active = {}
        for pair, force in forces.items():
            if force < self.minimum_force_n:
                continue
            start, peak = self.active.get(pair, (timestamp, 0.0))
            active[pair] = (start, max(peak, force))
            duration = timestamp - start + self.timestep
            if duration + 1e-9 >= self.minimum_seconds and pair not in self.reported:
                self.reported.add(pair)
                self.failures.append(dict(kind="unintended_contact", bodies=list(pair),
                                          sim_time=start, duration_at_detection_s=duration,
                                          peak_force_at_detection_n=active[pair][1]))
        self.active = active
