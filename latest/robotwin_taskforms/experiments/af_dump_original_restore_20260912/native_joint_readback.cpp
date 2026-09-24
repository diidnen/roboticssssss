// Read-only PhysX 5.3 engineering instrumentation. Not a replacement AF sensor.
// No callbacks, material changes, solver changes, or hidden friction reads.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sapien/physx/rigid_component.h>
#include <PxArticulationReducedCoordinate.h>
#include <PxArticulationLink.h>
#include <PxScene.h>
#include <PxPhysics.h>
#include <map>
#include <algorithm>
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
  result["com_linear_velocity_world"] = v(link->getLinearVelocity());
  result["angular_velocity_world"] = v(link->getAngularVelocity());
  auto com = link->getGlobalPose() * link->getCMassLocalPose();
  result["com_position_world"] = v(com.p);
  result["com_quaternion_world_wxyz"] = std::vector<float>{com.q.w, com.q.x, com.q.y, com.q.z};
  result["joint_quaternion_world_wxyz"] = std::vector<float>{frame.q.w, frame.q.x, frame.q.y, frame.q.z};
  result["mass_kg"] = link->getMass();
  result["linear_damping"] = link->getLinearDamping();
  result["angular_damping"] = link->getAngularDamping();
  result["max_linear_velocity"] = link->getMaxLinearVelocity();
  result["gravity_world"] = v(gravity);
  result["gravity_disabled"] = gravity_disabled;
  result["unqualified_contact_balance_world_n"] = v(contact_balance);
  result["source"] = "READ_ONLY_PHYSX_LINK_CACHE_NOT_YET_QUALIFIED_AS_AF_SENSOR";
  return result;
}

// Explicit diagnostic STATE MUTATION, never invoked by sensor readers.
// Canonicalize solver/contact history before restoring the same saved state
// for EVERY reference/candidate. No physics step or parameter changes here.
static py::dict cold_solver_reset(sapien::physx::PhysxRigidBaseComponent *component, bool flush = false,
                                  bool canonical_ids = false, bool canonical_shapes = false) {
  auto *scene = component->getPxActor()->getScene();
  if (!scene || scene->getNbAggregates()) throw std::runtime_error("Cold reset requires a scene without aggregates");
  const PxActorTypeFlags flags = PxActorTypeFlag::eRIGID_STATIC | PxActorTypeFlag::eRIGID_DYNAMIC;
  std::vector<PxActor*> actors(scene->getNbActors(flags));
  scene->getActors(flags, actors.data(), actors.size());
  std::vector<PxArticulationReducedCoordinate*> arts(scene->getNbArticulations());
  scene->getArticulations(arts.data(), arts.size());
  std::sort(actors.begin(),actors.end());
  std::sort(arts.begin(),arts.end());
  std::vector<PxU32> before_indices;
  for (auto *actor:actors) before_indices.push_back(actor->is<PxRigidActor>()->getInternalActorIndex());
  PxU32 slots = actors.size();
  for (auto *art:arts) slots += art->getNbLinks();
  PxU32 shape_slots = 0;
  for (auto *actor:actors) shape_slots += actor->is<PxRigidActor>()->getNbShapes();
  for (auto *art:arts) {
    std::vector<PxArticulationLink*> links(art->getNbLinks());
    art->getLinks(links.data(),links.size());
    for (auto *link:links) shape_slots += link->getNbShapes();
  }
  for (auto *art:arts) scene->removeArticulation(*art,false);
  if (!actors.empty()) scene->removeActors(actors.data(),actors.size(),false);
  if (flush) scene->flushSimulation(false);
  if (canonical_ids) {
    // Native actor IDs affect solver ordering. Removing/reinserting the same
    // objects in pointer order does NOT preserve IDs (v11 counterexample).
    // Canonicalize the now-empty scene's ID free-list with shape-less temporary
    // actors. They never participate in a physics step or touch a task object.
    std::vector<PxRigidStatic*> placeholders;
    PxMaterial *placeholder_material = nullptr;
    PxShape *placeholder_shape = nullptr;
    if (canonical_shapes) {
      placeholder_material = scene->getPhysics().createMaterial(0.f,0.f,0.f);
      placeholder_shape = scene->getPhysics().createShape(PxBoxGeometry(.001f,.001f,.001f),*placeholder_material);
    }
    const PxU32 count = canonical_shapes ? std::max(slots,shape_slots) : slots;
    for (PxU32 i=0;i<count;++i) {
      auto *actor = scene->getPhysics().createRigidStatic(PxTransform(PxVec3(1000.f+i*.1f,0.f,0.f)));
      if (!actor) throw std::runtime_error("ID canonicalization allocation failed");
      if (placeholder_shape) actor->attachShape(*placeholder_shape);
      scene->addActor(*actor);
      placeholders.push_back(actor);
    }
    std::sort(placeholders.begin(),placeholders.end(),[](auto *a,auto *b) {
      return a->getInternalActorIndex() > b->getInternalActorIndex();
    });
    for (auto *actor:placeholders) { scene->removeActor(*actor,false); actor->release(); }
    if (placeholder_shape) placeholder_shape->release();
    if (placeholder_material) placeholder_material->release();
    scene->flushSimulation(false);
  }
  if (!actors.empty() && !scene->addActors(actors.data(),actors.size())) throw std::runtime_error("Actor reinsertion failed");
  for (auto *art:arts) if (!scene->addArticulation(*art)) throw std::runtime_error("Articulation reinsertion failed");
  py::dict report;
  report["actors"] = actors.size(); report["articulations"] = arts.size();
  report["buffers_flushed_while_empty"] = flush;
  report["canonical_actor_id_free_list"] = canonical_ids;
  report["canonical_shape_allocation_requested"] = canonical_shapes;
  report["original_collision_shape_instances"] = shape_slots;
  report["actor_indices_before"] = before_indices;
  std::vector<PxU32> after_indices;
  for (auto *actor:actors) after_indices.push_back(actor->is<PxRigidActor>()->getInternalActorIndex());
  report["actor_indices_after"] = after_indices;
  report["scope"] = "EXPLICIT_COLD_SOLVER_RESET_REQUIRES_STATE_RESTORE_NOT_SENSOR_READ";
  return report;
}

// Explicit replay-boundary operation, never part of a sensor read.
static void update_kinematics(sapien::physx::PhysxRigidBaseComponent *component) {
  auto *link = component->getPxActor()->is<PxArticulationLink>();
  if (!link) throw std::runtime_error("Expected articulation link");
  link->getArticulation().updateKinematic(PxArticulationKinematicFlag::ePOSITION |
                                          PxArticulationKinematicFlag::eVELOCITY);
}

PYBIND11_MODULE(af_native_joint_readback, m) {
  m.def("read_link", &read_link);
  m.def("target_rays", &target_rays);
  m.def("cold_solver_reset", &cold_solver_reset, py::arg("component"), py::arg("flush") = false,
        py::arg("canonical_ids") = false, py::arg("canonical_shapes") = false);
  m.def("update_kinematics", &update_kinematics);
}
