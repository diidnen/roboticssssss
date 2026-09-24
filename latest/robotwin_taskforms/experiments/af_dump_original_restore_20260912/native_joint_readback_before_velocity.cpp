// Read-only PhysX 5.3 engineering instrumentation. Not a replacement AF sensor.
// No callbacks, material changes, solver changes, or hidden friction reads.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sapien/physx/rigid_component.h>
#include <PxArticulationReducedCoordinate.h>
#include <PxArticulationLink.h>
#include <PxScene.h>
#include <map>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace physx;

static std::vector<float> v(PxVec3 const &x) { return {x.x, x.y, x.z}; }

// Diagnostic process owns a single scene. SAPIEN hides the cache-release symbol;
// keep at most four caches until process exit rather than allocating per read.
// Do not use this provisional extension in a multi-scene production collector.
static std::map<PxArticulationReducedCoordinate *, PxArticulationCache *> caches;

struct TargetOnly : PxQueryFilterCallback {
  PxRigidActor *target;
  explicit TargetOnly(PxRigidActor *actor) : target(actor) {}
  PxQueryHitType::Enum preFilter(PxFilterData const &, PxShape const *, PxRigidActor const *actor, PxHitFlags &) override {
    return actor == target ? PxQueryHitType::eBLOCK : PxQueryHitType::eNONE;
  }
  PxQueryHitType::Enum postFilter(PxFilterData const &, PxQueryHit const &, PxShape const *, PxRigidActor const *actor) override {
    return actor == target ? PxQueryHitType::eBLOCK : PxQueryHitType::eNONE;
  }
};

static py::array_t<float> target_rays(sapien::physx::PhysxRigidBaseComponent *component,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> origins,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> directions,
                                    float distance) {
  auto o = origins.unchecked<2>(); auto d = directions.unchecked<2>();
  if (o.shape(1) != 3 || d.shape(1) != 3 || o.shape(0) != d.shape(0) || distance <= 0)
    throw std::runtime_error("Invalid ray arrays");
  auto *actor = component->getPxActor();
  TargetOnly filter(actor);
  PxQueryFilterData data(PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC | PxQueryFlag::ePREFILTER);
  py::array_t<float> result(o.shape(0)); auto out = result.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < o.shape(0); ++i) {
    PxVec3 origin(o(i,0), o(i,1), o(i,2)), direction(d(i,0), d(i,1), d(i,2));
    if (fabs(direction.magnitudeSquared() - 1.f) > 1e-4) throw std::runtime_error("Nonunit ray");
    PxRaycastBuffer hit;
    bool found = actor->getScene()->raycast(origin, direction, distance, hit, PxHitFlag::eDEFAULT, data, &filter);
    out(i) = found && hit.hasBlock ? hit.block.distance : -1.f;
  }
  return result;
}

static py::dict read_link(sapien::physx::PhysxRigidBaseComponent *component) {
  auto *link = component->getPxActor()->is<PxArticulationLink>();
  if (!link) throw std::runtime_error("Expected articulation link");
  auto &art = link->getArticulation();
  if (!caches.count(&art)) {
    if (caches.size() >= 4) throw std::runtime_error("Diagnostic cache limit exceeded");
    caches[&art] = art.createCache();
  }
  auto *cache = caches.at(&art);
  if (!cache) throw std::runtime_error("No articulation cache");
  art.copyInternalStateToCache(*cache, PxArticulationCacheFlag::eLINK_INCOMING_JOINT_FORCE |
                                      PxArticulationCacheFlag::eLINK_ACCELERATION);
  auto index = link->getLinkIndex();
  auto incoming = cache->linkIncomingJointForce[index];
  auto acceleration = cache->linkAcceleration[index];
  auto *joint = link->getInboundJoint();
  if (!joint) throw std::runtime_error("Root is not a finger");
  auto frame = link->getGlobalPose() * joint->getChildPose();
  auto force_world = frame.q.rotate(incoming.force);
  auto gravity = art.getScene()->getGravity();
  bool gravity_disabled = link->getActorFlags().isSet(PxActorFlag::eDISABLE_GRAVITY);
  if (gravity_disabled) gravity = PxVec3(0.f);
  auto contact_balance = acceleration.linear * link->getMass() - gravity * link->getMass() - force_world;
  py::dict result;
  result["link_index"] = index;
  result["joint_force_child_frame_n"] = v(incoming.force);
  result["joint_force_world_n"] = v(force_world);
  result["joint_torque_child_frame_nm"] = v(incoming.torque);
  result["com_linear_acceleration"] = v(acceleration.linear);
  result["com_angular_acceleration"] = v(acceleration.angular);
  result["mass_kg"] = link->getMass();
  result["gravity_world"] = v(gravity);
  result["gravity_disabled"] = gravity_disabled;
  result["unqualified_contact_balance_world_n"] = v(contact_balance);
  result["source"] = "READ_ONLY_PHYSX_LINK_CACHE_NOT_YET_QUALIFIED_AS_AF_SENSOR";
  return result;
}

PYBIND11_MODULE(af_native_joint_readback, m) {
  m.def("read_link", &read_link);
  m.def("target_rays", &target_rays);
}
