import copy
import math
import random

from htn_planner import HTNPlanner
from utils import (
	manhattan, has_line_of_sight, under_friendly_drone_cover, next_step,
	get_num_attacks, get_penetration_probability, perform_attack,
	all_units_at_position, compute_staging_position
)
from terrain_utils import sign
from log import logger


class FriendlyUnit:
	"""Base class for friendly units using HTN-based planning and reactive combat logic."""

	def __init__(self, name, state, domain, simulation=None):
		"""Initialize a friendly unit with planning state and simulation context."""
		self.name = name
		self.state = state
		self.planner = HTNPlanner(domain)
		self.current_plan = []
		self.sim = simulation
		self.last_position = state["position"]
		self.last_health = state["health"]
		self.last_group_size = state["current_group_size"]

	def update_plan(self, force_replan=False):
		"""Regenerate or preserve the current plan depending on context or threats."""
		mission = "SecureOutpostMission"

		# ─── only broadcast once we're about to execute AttackEnemy ─────────────
		if not force_replan and self.current_plan:
			first = self.current_plan[0]
			if isinstance(first, tuple) and first[0] == "AttackEnemy":
				enemy_name = first[1]
				enemy = self.sim.enemy_units_dict.get(enemy_name)
				if enemy and self.can_attack(enemy):
					logger.info(f"{self.name} next target {enemy_name} in range; broadcasting replan")
					for u in self.sim.friendly_units:
						if u is not self:
							u.update_plan(force_replan=True)
					force_replan = True
		# ─────────────────────────────────────────────────────────────────────────

		if force_replan or not self.current_plan:
			combined = copy.deepcopy(self.state)
			combined.update({
				"sim":               self.sim,
				"unit":              self,
				"enemy_units_dict":  self.sim.enemy_units_dict,
				"spotted_enemies":   [
					name for name in self.state.get("spotted_enemies", [])
					if name in self.sim.enemy_units_dict
					and self.sim.enemy_units_dict[name].state.get("enemy_alive", False)
				]
			})
			self.current_plan = self.planner.plan(mission, combined) or [("Hold", None)]
			logger.info(f"{self.name} replanned: {self.current_plan}")
		else:
			logger.info(f"{self.name} current plan: {self.current_plan}")

		self.last_health = self.state["health"]


	def execute_next_task(self):
		"""Execute the next action in the HTN plan."""
		logger.info(f"{self.name} is facing {self.state['facing']}")
		if not self.current_plan or self.state["health"] <= 0:
			logger.info(f"{self.name} cannot execute task: plan empty or health <= 0")
			return

		task_name, task_arg = self.current_plan[0] if isinstance(self.current_plan[0], tuple) else (self.current_plan[0], None)
		logger.info(f"{self.name} executing task: {task_name}, arg: {task_arg}")

		if task_name == "Move":
			self._execute_move(task_arg)
		elif task_name == "AttackEnemy":
			self._execute_attack(task_arg)
		elif task_name == "SecureOutpostNoArg":
			self._execute_secure_outpost()
		elif task_name == "MoveToStaging":
			self._execute_move_to_staging()
		elif task_name == "WaitForGroup":
			self._execute_wait_for_group()
		elif task_name == "Hold":
			logger.info(f"{self.name} holds position at {self.state['position']}")
			self.current_plan.pop(0)

	def _execute_move(self, target_name):
		"""
		Move toward the named target, which may be an enemy unit or the 'outpost'.
		"""
		# 0) Resolve the target position
		if target_name == "outpost":
			target_pos = self.state.get("outpost_position")
			if target_pos is None:
				logger.info(f"{self.name} has no outpost_position; dropping Move('outpost')")
				self.current_plan.pop(0)
				return
			enemy = None
		else:
			enemy = self.sim.enemy_units_dict.get(target_name)
			if enemy is None or not enemy.state.get("enemy_alive", False):
				logger.info(f"{self.name} cannot Move→{target_name}: target gone; dropping step")
				self.current_plan.pop(0)
				self.update_plan(force_replan=True)
				return
			# 1) If in range & LOS for an enemy, switch to AttackEnemy
			if self.can_attack(enemy):
				logger.info(f"{self.name} is now in range/LOS of {target_name}, switching to AttackEnemy")
				self.current_plan[0] = ("AttackEnemy", target_name)
				return
			target_pos = enemy.state["position"]

		# 2) Move‐credit and stepping
		self.state["move_credit"] += self.state["speed"]
		steps = int(self.state["move_credit"])
		self.state["move_credit"] -= steps
		logger.info(f"{self.name} taking {steps} steps toward {target_name}")

		for _ in range(steps):
			old = self.state["position"]
			if old == target_pos:
				logger.info(f"{self.name} reached {target_name} at {target_pos}")
				self.current_plan.pop(0)
				break

			new = next_step(old, target_pos, self.sim.enemy_units, unit=self.state["type"])
			self.state["position"] = new
			dx, dy = new[0] - old[0], new[1] - old[1]
			if dx or dy:
				self.state["facing"] = (sign(dx), sign(dy))
			logger.info(f"{self.name} moved to {new}")


	def _execute_attack(self, enemy_name):
		"""Attempt to attack the specified enemy unit; fall back to Move if needed."""
		target = self._get_enemy_by_name(enemy_name)
		# 1) Dead or missing?  Pop and replan globally (to pick a new target/mission).
		if not target or not target.state.get("enemy_alive", False):
			logger.info(f"{self.name} cannot attack; target {enemy_name} is dead or missing.")
			self.current_plan.pop(0)
			self.update_plan(force_replan=True)
			return

		# 2) Out of range or no LOS?  Swap this AttackEnemy into a Move step only.
		if not self.can_attack(target):
			logger.info(f"{self.name} cannot attack {target.name} yet; switching to Move.")
			# Replace the pending AttackEnemy with Move toward the same enemy
			self.current_plan[0] = ("Move", enemy_name)
			return

		# 3) Otherwise, we’re good—perform the shot.
		perform_attack(self, target)

		# 4) If we killed it, pop the attack and force everyone to replan
		if not target.state.get("enemy_alive", False):
			self.current_plan.pop(0)
			for u in self.sim.friendly_units:
				u.update_plan(force_replan=True)


	def _execute_secure_outpost(self):
		"""Secure the outpost if at correct position."""
		if self.state["position"] == self.state["outpost_position"]:
			self.state["outpost_secured"] = True
			logger.info(f"{self.name} secures the outpost!")
		else:
			logger.info(f"{self.name} cannot secure outpost; not at target location.")
		self.current_plan.pop(0)
		self.current_plan.append(("Hold", None))

	def _execute_move_to_staging(self):
		"""Move toward the designated staging position."""

		# ─── STEP 0: CLEAR ANY PREVIOUS STAGING METADATA ────────────────
		# If we have an old 'staging_position' or 'all_arrived_flag' from last time,
		# remove it so that the next lines compute a fresh tile and force a new
		# “wait one extra round” when we enter WaitForGroup.
		self.state.pop("staging_position", None)
		self.state.pop("all_arrived_flag",    None)
		# ────────────────────────────────────────────────────────────────

		# (a) Accumulate movement credit and compute how many discrete steps this tick
		self.state["move_credit"] += self.state["speed"]
		steps = int(self.state["move_credit"])
		self.state["move_credit"] -= steps
		logger.info(f"{self.name} moving to staging area with {steps} steps")

		# (b) Compute (and store) a brand‐new staging_position every time we re‐enter MoveToStaging
		self.state["staging_position"] = compute_staging_position(self.sim)
		staging = self.state["staging_position"]

		# (c) Spend up to 'steps' moving toward that staging tile
		for _ in range(steps):
			old_pos = self.state["position"]
			if old_pos == staging:
				logger.info(f"{self.name} reached staging area {staging}")
				self.current_plan.pop(0)
				break

			self.state["position"] = next_step(
				old_pos,
				staging,
				self.sim.enemy_units,
				unit=self.state["type"]
			)

			dx, dy = self.state["position"][0] - old_pos[0], self.state["position"][1] - old_pos[1]
			if dx or dy:
				self.state["facing"] = (sign(dx), sign(dy))
			logger.info(f"{self.name} moved to {self.state['position']}")


	def _execute_wait_for_group(self):
		"""
		Hold until all *relevant* friendly units still doing this staging step have arrived,
		then wait one extra round (if desired), and finally pop out.
		"""

		# 1) Make sure we have a staging_position
		if "staging_position" not in self.state:
			self.state["staging_position"] = compute_staging_position(self.sim)
		staging = self.state["staging_position"]

		# 2) Build a list of “relevant buddies”: only those friendlies who
		#    (a) are still on MoveToStaging or WaitForGroup, and
		#    (b) share exactly this same staging_position.
		relevant_buddies = []
		for u in self.sim.friendly_units:
			if not u.current_plan:
				continue

			first = u.current_plan[0] if isinstance(u.current_plan[0], str) else u.current_plan[0][0]
			if first in ("MoveToStaging", "WaitForGroup"):
				# They need to be heading to—or still waiting at—the same staging tile
				if u.state.get("staging_position", None) == staging:
					relevant_buddies.append(u)

		# 3) If any of those “relevant buddies” are not yet standing on 'staging',
		#    then we keep waiting.
		all_those_here = all((u.state["position"] == staging) for u in relevant_buddies)
		if not all_those_here:
			logger.info(f"{self.name} waiting for {len(relevant_buddies)} buddies at {staging}")
			return

		# 4) Optional: if you still want the “one extra round after everyone arrives” behavior,
		#    you can use an 'all_arrived_flag' as before. If you just want to pop out immediately,
		#    skip this block and always pop below.
		if not self.state.get("all_arrived_flag", False):
			self.state["all_arrived_flag"] = True
			logger.info(f"{self.name} all relevant units here; waiting one extra round")
			return

		self.state["staging_complete"] = True
		
		# 5) Everyone who needed to stage here has arrived, and we’ve done the extra tick.
		logger.info(f"{self.name} staging complete; proceeding from {staging}")
		self.state.pop("all_arrived_flag", None)
		self.current_plan.pop(0)


	def _get_enemy_by_name(self, name):
		"""Return enemy unit by name from the simulation context."""
		return next((e for e in self.sim.enemy_units if e.name == name), None)

	def _is_about_to_attack(self, target_arg):
		"""Check if the next task is an attack on the specified target."""
		return (
			len(self.current_plan) > 1
			and isinstance(self.current_plan[1], tuple)
			and self.current_plan[1][0] == "AttackEnemy"
			and target_arg == self.current_plan[1][1]
		)

	def get_goal_position(self, task=None):
		"""Determine movement goal based on current task and drone intelligence."""
		task_name, task_arg = task if isinstance(task, tuple) else (task, None)
		if task_name == "Move" and task_arg == "outpost":
			return self.state["outpost_position"]
		if task_name == "MoveToStaging":
			if "staging_position" not in self.state:
				self.state["staging_position"] = compute_staging_position(self.sim)
			return self.state["staging_position"]
		if task_name in ("Move", "AttackEnemy") and task_arg:
			return self.sim.friendly_drone.last_known.get(task_arg, self.state["position"])
		return self.state["position"]

	def needs_update(self):
		"""Determine whether the unit has changed enough to require replanning."""
		return (
			self.state["position"] != self.last_position
			or abs(self.state["health"] - self.last_health) > 0.1
			or self.state["current_group_size"] != self.last_group_size
			or self.state["health"] <= 0
		)


class FriendlyTank(FriendlyUnit):
	"""Tank unit with line-of-sight and range-based attack logic."""
	def can_attack(self, target):
		return target.state.get("enemy_alive", False) and has_line_of_sight(self.state["position"], target.state["position"]) and manhattan(self.state["position"], target.state["position"]) <= self.state["attack_range"]


class FriendlyInfantry(FriendlyTank):
	"""Infantry unit reusing attack logic from tank."""
	pass


class FriendlyAntiTank(FriendlyTank):
	"""Anti-tank unit reusing attack logic from tank."""
	pass


class FriendlyScout(FriendlyUnit):
	"""Scout unit with custom behavior (to be defined)."""
	pass


class FriendlyArtillery(FriendlyUnit):
	"""Artillery unit with support-enabled attack capability."""
	def can_attack(self, target):
		if not target.state.get("enemy_alive", False):
			return False

		distance = manhattan(self.state["position"], target.state["position"])
		if distance > self.state["attack_range"]:
			return False

		if has_line_of_sight(self.state["position"], target.state["position"]):
			return True

		for unit in self.sim.friendly_units:
			if unit is not self and has_line_of_sight(unit.state["position"], target.state["position"]):
				return True

		return under_friendly_drone_cover(self.sim, target.state["position"])
