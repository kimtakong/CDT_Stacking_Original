import os
import hashlib
import colorsys
import math
from typing import Optional, Tuple, Any
import pandas as pd

try:
	from direct.showbase.ShowBase import ShowBase
	from direct.task import Task
	from panda3d.core import (
		WindowProperties, 
		GeomNode, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
		Geom, GeomTriangles, GeomLines,
		NodePath, LVector3,
		AmbientLight, DirectionalLight,
		TextNode,
	)
except Exception as e:
	ShowBase = None
	print(f"Warning: Panda3D not available: {e}")

from .config import YardSimConfig
from .core import YardSimulation, BaseStackingStrategy



# ======================== 3D Panda3D Renderer ======================== #
class Panda3DYardRenderer(ShowBase):
	"""3D rendering of the container yard using Panda3D.

	Draws 3D boxes for containers at (block, x, y, z). 40ft containers
	are rendered as two adjacent cubes along +x. GP/RF colored differently.
	"""
	def __init__(
		self,
		yard: YardSimulation,
		config: YardSimConfig,
		blocks_per_row: int = 2,
		lane_every: int = 5,
		colorful: bool = True,
		hud_title: str = "",
		window_title: str = "Container Yard 3D",
		window_size: Tuple[int, int] = (1280, 720),
	) -> None:
		if ShowBase is None:
			raise ImportError("Panda3D is required for 3D visualization. Install with `pip install panda3d`.")

		ShowBase.__init__(self)
		
		self.yard = yard
		self.cfg = config or getattr(yard, "yard_config", YardSimConfig())
		self.blocks_per_row = max(1, int(blocks_per_row))
		self.lane_every = max(0, int(lane_every))
		
		# Calculate total blocks to display (max of GP and RF blocks)
		# Handle case where config might not have gp_blocks/rf_blocks (backward compatibility)
		self.gp_blocks = getattr(self.cfg, 'gp_blocks', getattr(self.cfg, 'blocks', 3))
		self.rf_blocks = getattr(self.cfg, 'rf_blocks', 1) # Default to 1 if not present
		self.total_blocks = max(self.gp_blocks, self.rf_blocks)
		self.colorful = colorful
		self.hud_title = hud_title or ""
		
		# Configure window
		props = WindowProperties()
		props.setTitle(window_title)
		props.setSize(window_size[0], window_size[1])
		self.win.requestProperties(props)
		
		# Colors (normalized to 0-1 range for Panda3D)
		self.bg_color = (0.9, 0.9, 0.9, 1.0)
		self.default_color = (0.7, 0.7, 0.7, 1.0)  # Default gray for all containers
		self.gp_base = (60/255, 170/255, 230/255, 1.0)
		self.rf_base = (230/255, 130/255, 70/255, 1.0)
		self.in_color = (0.2, 0.8, 0.2, 1.0)  # Bright green for IN events
		self.update_color = (0.2, 0.5, 1.0, 1.0)  # Blue for CUS/COR/COP events
		self.remove_color = (0.9, 0.2, 0.2, 1.0)  # Red for remove events
		self.rehandling_color = (0.9, 0.7, 0.2, 1.0)  # Orange/Yellow for rehandling
		self.overdue_color = (0.6, 0.2, 0.8, 1.0)  # Purple for overdue containers
		self.grid_color = (0.5, 0.5, 0.5, 1.0)
		self.lane_color = (0.7, 0.7, 0.7, 1.0)
		self.plug_color = (1.0, 165/255, 0.0, 1.0)
		
		# Event tracking: {container_key: {'event_type': str, 'frame_count': int, 'duration_frames': int}}
		self.container_events = {}
		self.event_frame_counter = 0  # Global frame counter for events
		
		# Container NodePath tracking: {container_key: NodePath}
		self.container_nodes = {}
		
		# Animation queue for rehandling and removal
		self.animation_queue = []
		self.is_animating = False
		
		# Current simulation time for dwell time calculation
		self.current_sim_time = None
		
		# Set background color
		self.setBackgroundColor(*self.bg_color)
		
		# Container node to hold all yard geometry
		self.yard_root = self.render.attachNewNode("yard_root")
		# Move yard inland (further from the water) - increased distance to avoid overlap with QC
		# QC landside legs are at y=20 (base y=5 + 15), so yard needs to start after y=25
		self.yard_root.setY(150)  # Move 60 units inland to ensure safe distance from QC
		
		# Create separate root for port elements (ships, QC, ocean) that should stay at waterfront
		self.port_root = self.render.attachNewNode("port_root")
		# Port elements stay at y=0 (waterfront)
		
		# Create separate root for decorations (fence, lighting, office, gate) that can be toggled
		self.decorations_root = self.render.attachNewNode("decorations_root")
		
		# Create separate root for yard labels (GP0, RF0, etc.) that can be toggled
		self.yard_labels_root = self.render.attachNewNode("yard_labels_root")
		
		# Toggle state for decorations and text visibility
		self.show_decorations_and_text = True
		
		# Camera setup (isometric-like view)
		self.disableMouse()
		
		# Calculate layout dimensions for rows of 10 blocks
		BLOCKS_PER_ROW = 10
		total_rows = (self.total_blocks + BLOCKS_PER_ROW - 1) // BLOCKS_PER_ROW  # Ceiling division
		layout_width = min(self.total_blocks, BLOCKS_PER_ROW) * (self.cfg.width + 5)
		# Each row has GP + RF yards with spacing
		layout_height = total_rows * ((self.cfg.height + 5) * 2 + 10)
		
		self.camera_distance = max(layout_width, layout_height) * 1.2
		self.camera_yaw = 45
		self.camera_pitch = -45
		
		# Camera focus point (center of the entire layout)
		center_x = layout_width / 2
		center_y = layout_height / 2
		center_z = self.cfg.depth / 2
		self.camera_focus = [center_x, center_y, center_z]
		self._update_camera_position()
		
		# Lighting
		self._setup_lighting()
		
		# Create ground grid and containers
		self._build_scene()
		
		# HUD text
		self._setup_hud()
		
		# Mouse control state
		self.mouse_down = False
		self.last_mouse_x = 0
		self.last_mouse_y = 0
		self.click_start_pos = None  # Track click start position to distinguish click from drag
		
		# Accept input events
		self.accept("escape", self.user_exit)
		self.accept("mouse1", self._on_mouse_down)
		self.accept("mouse1-up", self._on_mouse_up)
		self.accept("wheel_up", self._on_zoom_in)
		self.accept("wheel_down", self._on_zoom_out)
		self.accept("arrow_left", self._rotate_left)
		self.accept("arrow_right", self._rotate_right)
		self.accept("arrow_up", self._rotate_up)
		self.accept("arrow_down", self._rotate_down)
		self.accept("a", self._rotate_left)
		self.accept("d", self._rotate_right)
		self.accept("w", self._rotate_up)
		self.accept("s", self._rotate_down)
		
		# Ctrl + Click to change focus point
		self.accept("control-mouse1", self._on_ctrl_click)
		
		# View presets (1-4 keys for different viewing angles)
		self.accept("1", lambda: self._set_preset_view("NE"))
		self.accept("2", lambda: self._set_preset_view("SE"))
		self.accept("3", lambda: self._set_preset_view("SW"))
		self.accept("4", lambda: self._set_preset_view("NW"))
		self.accept("5", lambda: self._set_preset_view("TOP"))
		self.accept("6", lambda: self._set_preset_view("FRONT"))
		
		# Zoom presets
		self.accept("r", self._reset_view)  # Reset to default view
		self.accept("=", self._on_zoom_in)  # Alternative zoom in
		self.accept("-", self._on_zoom_out)  # Alternative zoom out
		self.accept("+", self._on_zoom_in)  # Alternative zoom in
		self.accept("_", self._on_zoom_out)  # Alternative zoom out
		
		# EDI Color Mode
		self.edi_color_mode = False
		self.accept("c", self._toggle_edi_color_mode)
		
		# Toggle decorations and text (t key)
		self.accept("t", self._toggle_decorations_and_text)
		
		# EDI Colors
		self.edi_colors = {
			'CUS': (0.53, 0.81, 0.92, 1.0),  # Sky Blue
			'COR': (0.0, 0.0, 1.0, 1.0),     # Blue
			'COP': (1.0, 0.0, 0.0, 1.0),     # Red
			'OUT': (0.1, 0.1, 0.1, 1.0),     # Black
			'NONE': (0.7, 0.7, 0.7, 1.0)     # Gray (Default)
		}
		
		# Task for mouse drag handling
		self.taskMgr.add(self._mouse_task, "mouse_task")
		self.taskMgr.add(self._update_hud_task, "hud_task")
		self.taskMgr.add(self._event_update_task, "event_update_task")
	
	def _setup_lighting(self):
		"""Setup ambient and directional lighting."""
		# Ambient light
		alight = AmbientLight("ambient")
		alight.setColor((0.4, 0.4, 0.4, 1))
		alnp = self.render.attachNewNode(alight)
		self.render.setLight(alnp)
		
		# Directional light (sun)
		dlight = DirectionalLight("directional")
		dlight.setColor((0.8, 0.8, 0.8, 1))
		dlnp = self.render.attachNewNode(dlight)
		dlnp.setHpr(45, -60, 0)
		self.render.setLight(dlnp)
	
	def _setup_hud(self):
		"""Setup HUD text for displaying rehandling count and next container info."""
		# Load Korean font for all HUD elements
		korean_font = None
		try:
			# Try to load KoPub Batang Medium from project folder
			korean_font = self.loader.loadFont('KoPub Batang Medium.ttf')
			print("Successfully loaded KoPub Batang Medium font")
		except:
			try:
				# Fallback to system fonts
				korean_font = self.loader.loadFont('c:/Windows/Fonts/malgun.ttf')
			except:
				print("Warning: Could not load Korean font. Korean characters may not display correctly.")
		
		# Main HUD (rehandling count) - top right
		self.hud_text = TextNode("hud")
		self.hud_text.setText("")
		if korean_font:
			self.hud_text.setFont(korean_font)
		self.hud_text.setTextColor(1, 0.65, 0, 1)  # Orange
		self.hud_text.setAlign(TextNode.ARight)
		self.hud_np = self.aspect2d.attachNewNode(self.hud_text)
		self.hud_np.setScale(0.07)
		self.hud_np.setPos(1.3, 0, 0.9)
		
		# Next container info HUD (top-left, larger and more left)
		self.next_container_text = TextNode("next_container")
		self.next_container_text.setText("")
		if korean_font:
			self.next_container_text.setFont(korean_font)
		self.next_container_text.setTextColor(0.2, 0.8, 0.2, 1)  # Green
		self.next_container_text.setAlign(TextNode.ALeft)
		self.next_container_np = self.aspect2d.attachNewNode(self.next_container_text)
		self.next_container_np.setScale(0.08)  # Larger font
		self.next_container_np.setPos(-1.6, 0, 0.9)  # More to the left
		
		# Current simulation time HUD (bottom-right)
		self.sim_time_text = TextNode("sim_time")
		self.sim_time_text.setText("--")
		if korean_font:
			self.sim_time_text.setFont(korean_font)
		self.sim_time_text.setTextColor(0, 0, 0, 1)  # Black
		self.sim_time_text.setAlign(TextNode.ARight)
		self.sim_time_np = self.aspect2d.attachNewNode(self.sim_time_text)
		self.sim_time_np.setScale(0.10)  # Larger font
		self.sim_time_np.setPos(1.3, 0, -0.9)  # Bottom right
		
		# Container info panel (bottom-left, initially hidden)
		self.container_info_text = TextNode("container_info")
		self.container_info_text.setText("")
		if korean_font:
			self.container_info_text.setFont(korean_font)
		self.container_info_text.setTextColor(0, 0, 0, 1)  # Yellow (more visible)
		self.container_info_text.setAlign(TextNode.ALeft)
		self.container_info_text.setShadow(0.05, 0.05)  # Add shadow for better visibility
		self.container_info_text.setShadowColor(0, 0, 0, 1)  # Black shadow
		self.container_info_np = self.aspect2d.attachNewNode(self.container_info_text)
		self.container_info_np.setScale(0.04)  # Smaller font size
		self.container_info_np.setPos(-1.6, 0, -0.3)  # Higher position
		
		# Make sure text is rendered properly
		self.container_info_np.setBin("fixed", 1)
		self.container_info_np.setDepthTest(False)
		self.container_info_np.setDepthWrite(False)
		
		self.container_info_np.hide()  # Initially hidden
		
		# Camera position HUD (top-right, below hud_text)
		self.camera_pos_text = TextNode("camera_pos")
		self.camera_pos_text.setText("")
		if korean_font:
			self.camera_pos_text.setFont(korean_font)
		self.camera_pos_text.setTextColor(0.2, 0.2, 0.8, 1)  # Blue
		self.camera_pos_text.setAlign(TextNode.ARight)  # Right align
		self.camera_pos_np = self.aspect2d.attachNewNode(self.camera_pos_text)
		self.camera_pos_np.setScale(0.05)
		self.camera_pos_np.setPos(1.3, 0, 0.75)  # Top-right, below hud_text
		
		# Selected container tracking
		self.selected_container_key = None
	
	def _update_hud_task(self, task):
		"""Update HUD text periodically."""
		try:
			status = self.yard.get_yard_status() or {}
			rh_count = int(status.get('rehandling_count', getattr(self.yard, 'rehandling_count', 0) or 0))
			text = f"Rehandling: {rh_count}"
			if self.hud_title:
				text = f"{text} | {self.hud_title}"
			self.hud_text.setText(text)
			
			# Update camera position text
			cam_pos = self.camera.getPos()
			focus = self.camera_focus
			self.camera_pos_text.setText(
				f"Camera: ({cam_pos.x:.1f}, {cam_pos.y:.1f}, {cam_pos.z:.1f})\n"
				f"Focus: ({focus[0]:.1f}, {focus[1]:.1f}, {focus[2]:.1f})\n"
				f"Distance: {self.camera_distance:.1f} | Yaw: {self.camera_yaw:.1f} | Pitch: {self.camera_pitch:.1f}"
			)
		except Exception as e:
			pass
		return Task.cont
	
	def _update_camera_position(self):
		"""Update camera position based on yaw, pitch, and distance."""
		rad_yaw = math.radians(self.camera_yaw)
		rad_pitch = math.radians(self.camera_pitch)
		
		# Calculate camera position in spherical coordinates around focus point
		x = self.camera_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
		y = self.camera_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
		z = self.camera_distance * math.sin(rad_pitch)
		
		# Position camera relative to focus point
		self.camera.setPos(
			self.camera_focus[0] + x,
			self.camera_focus[1] + y,
			self.camera_focus[2] + z
		)
		self.camera.lookAt(self.camera_focus[0], self.camera_focus[1], self.camera_focus[2])
	
	# def _color_for_key(self, key: str, fallback: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
	# 	"""Return color for a container based on events and dwell time. Default is gray unless there's an active event."""
		
	# 	# EDI Color Mode Logic
	# 	if self.edi_color_mode:
	# 		container_info = self.yard.containers.get(key)
	# 		base_color = self.edi_colors['NONE']
	# 		if container_info:
	# 			# Check event history for the latest EDI status
	# 			event_history = container_info.get('event_history', [])
	# 			last_event = event_history[-1] if event_history else 'NONE'
				
	# 			if last_event == 'IN':
	# 				base_color = self.edi_colors['NONE']
	# 			elif last_event in self.edi_colors:
	# 				base_color = self.edi_colors[last_event]
	# 			elif last_event == 'OUT':
	# 				base_color = self.edi_colors['OUT']

	# 			brightness, _, _ = self._compute_actual_cdt_brightness(container_info)
	# 			if brightness is not None:
	# 				return self._scale_color(base_color, brightness)
	# 		return base_color

	# 	# Check if this container has an active event (highest priority)
	# 	if key in self.container_events:
	# 		event_info = self.container_events[key]
	# 		frames_elapsed = self.event_frame_counter - event_info['frame_count']
			
	# 		# If event duration (in frames) has passed, remove from tracking
	# 		if frames_elapsed >= event_info['duration_frames']:
	# 			del self.container_events[key]
	# 		else:
	# 			# Return event-specific color (takes precedence over dwell time)
	# 			event_type = event_info['event_type']
	# 			if event_type == 'in':
	# 				return self.in_color
	# 			elif event_type == 'update':
	# 				return self.update_color
	# 			elif event_type == 'remove':
	# 				return self.remove_color
	# 			elif event_type == 'rehandling':
	# 				return self.rehandling_color
		
	# 	# Check dwell time if no active event
	# 	if self.current_sim_time is not None:
	# 		container_info = self.yard.containers.get(key)
	# 		if container_info:
	# 			try:
	# 				from datetime import datetime, timedelta

	# 				cargo_type = container_info.get('CARGO_TYPE', '')
	# 				sztp = str(container_info.get('SZTP2', ''))
	# 				is_rf = cargo_type == 'RF' or 'R' in sztp
	# 				base_color = self.rf_base if is_rf else self.gp_base

	# 				brightness, current_time, in_time = self._compute_actual_cdt_brightness(container_info)
	# 				if brightness is not None:
	# 					return self._scale_color(base_color, brightness)

	# 				if current_time is not None and in_time is not None:
	# 					cdt_val = container_info.get('cdt_true') or container_info.get('CDT_true')
	# 					if cdt_val is None:
	# 						for k in ['std_CDT_pred', 'no_std_CDT_pred']:
	# 							if container_info.get(k) is not None:
	# 								cdt_val = container_info[k]
	# 								break

	# 					if cdt_val is not None:
	# 						total_duration = 0.0
	# 						remaining_duration = 0.0
							
	# 						is_timestamp = False
	# 						duration_hours = None
	# 						if isinstance(cdt_val, str):
	# 							try:
	# 								if 'T' in cdt_val or '-' in cdt_val or ':' in cdt_val:
	# 									datetime.fromisoformat(cdt_val.replace('Z', '+00:00'))
	# 									is_timestamp = True
	# 								else:
	# 									duration_hours = float(cdt_val)
	# 							except:
	# 								pass
	# 						elif isinstance(cdt_val, datetime):
	# 							is_timestamp = True
	# 						elif isinstance(cdt_val, timedelta):
	# 							duration_hours = cdt_val.total_seconds() / 3600.0
	# 						elif isinstance(cdt_val, (int, float)):
	# 							duration_hours = float(cdt_val)
							
	# 						if is_timestamp:
	# 							if isinstance(cdt_val, datetime):
	# 								departure_time = cdt_val
	# 							else:
	# 								departure_time = datetime.fromisoformat(str(cdt_val).replace('Z', '+00:00'))
	# 							total_duration = (departure_time - in_time).total_seconds() / 3600.0
	# 							remaining_duration = (departure_time - current_time).total_seconds() / 3600.0
	# 						elif duration_hours is not None and duration_hours > 0:
	# 							departure_time = in_time + timedelta(hours=duration_hours)
	# 							remaining_duration = (departure_time - current_time).total_seconds() / 3600.0
	# 							total_duration = duration_hours

	# 						if total_duration > 0:
	# 							factor = remaining_duration / total_duration
	# 							factor = max(0.0, min(1.0, factor))
	# 							r = base_color[0] * factor
	# 							g = base_color[1] * factor
	# 							b = base_color[2] * factor
	# 							return (r, g, b, 1.0)

	# 			except Exception as e:
	# 				pass
		
	# 	# Default gray color for all containers
	# 	return self.default_color

	# def _color_for_key(self, key: str, fallback: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
	# 		"""
	# 		[?섏젙?? CDT???곕씪 ?됱긽(Hue)??蹂寃쏀븯???덊듃留?諛⑹떇 (Blue -> Red)
	# 		- 吏㏃? 泥대쪟(0??: ?뚮???(Blue)
	# 		- 以묎컙 泥대쪟(2??: 珥덈줉???몃???(Green/Yellow)
	# 		- 湲?泥대쪟(4???댁긽): 鍮④컙??(Red)
	# 		"""
			
	# 		# 1. ?대깽??IN, OUT ??媛 ?덈뒗 寃쎌슦 理쒖슦?좎쑝濡??대떦 ?됱긽 ?쒖떆
	# 		if key in self.container_events:
	# 			event_info = self.container_events[key]
	# 			frames_elapsed = self.event_frame_counter - event_info['frame_count']
				
	# 			if frames_elapsed < event_info['duration_frames']:
	# 				event_type = event_info['event_type']
	# 				if event_type == 'in': return self.in_color
	# 				elif event_type == 'update': return self.update_color
	# 				elif event_type == 'remove': return self.remove_color
	# 				elif event_type == 'rehandling': return self.rehandling_color
	# 			else:
	# 				del self.container_events[key]

	# 		# 2. CDT 湲곕컲 ?덊듃留??됱긽 怨꾩궛
	# 		if self.current_sim_time is not None:
	# 			container_info = self.yard.containers.get(key)
	# 			if container_info:
	# 				try:
	# 					from datetime import datetime, timedelta

	# 					# --- [媛?媛?몄삤湲?濡쒖쭅: Predict ?곗꽑 -> Real] ---
	# 					# 1?쒖쐞: std_CDT_pred (?덉륫媛?
	# 					# 2?쒖쐞: no_std_CDT_pred
	# 					# 3?쒖쐞: CDT_true (?ㅼ젣媛?
	# 					target_val = container_info.get('std_CDT_pred') or container_info.get('no_std_CDT_pred') or container_info.get('CDT_true')
						
	# 					if target_val is not None:
	# 						# ??Day) ?⑥쐞濡?蹂??
	# 						cdt_days = 0.0
							
	# 						# 諛섏엯 ?쒓컙 怨꾩궛 (寃쎄낵 ?쒓컙 ?뺤씤??
	# 						in_time = None
	# 						if 'EVENT_TS' in container_info:
	# 							val = container_info['EVENT_TS']
	# 							if isinstance(val, str):
	# 								in_time = datetime.fromisoformat(val.replace('Z', '+00:00'))
	# 							else:
	# 								in_time = val

	# 						# CDT 媛??뚯떛
	# 						if isinstance(target_val, (int, float)):
	# 							cdt_days = float(target_val) / 24.0
	# 						elif isinstance(target_val, str):
	# 							try:
	# 								cdt_days = float(target_val) / 24.0
	# 							except: # Timestamp 臾몄옄?댁씤 寃쎌슦
	# 								if in_time:
	# 									dept = datetime.fromisoformat(target_val.replace('Z', '+00:00'))
	# 									cdt_days = (dept - in_time).total_seconds() / 86400.0
							
	# 						# --- [?듭떖: ?덊듃留?而щ윭 ?앹꽦] ---
	# 						# 湲곗?: 0??Blue) ~ 5??Red) 踰붿쐞濡??뺢퇋??
	# 						# 4???댁긽?대㈃ 臾댁“嫄?鍮④컙?됱쑝濡?怨좎젙
	# 						max_days = 5.0 
	# 						norm = max(0.0, min(1.0, cdt_days / max_days))
							
	# 						# HSV ?됯났媛??ъ슜: 
	# 						# Hue 0.66 (?뚮옉) -> Hue 0.33 (珥덈줉) -> Hue 0.0 (鍮④컯)
	# 						hue = 0.66 * (1.0 - norm) 
							
	# 						# HSV -> RGB 蹂??
	# 						r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0) # Saturation=1.0, Value=1.0 (?좊챸?섍쾶)
							
	# 						return (r, g, b, 1.0)

	# 				except Exception as e:
	# 					pass
			
	# 		# 湲곕낯媛?(?곗씠?곌? ?녾굅???먮윭 ??GP/RF 湲곕낯???ъ슜)
	# 		return fallback
	
	def _color_for_key(self, key: str, fallback: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
				"""
				[吏꾨떒 紐⑤뱶] ??쟾 ?곸옱(Inverse Stacking) ?섏씠?쇱씠??- Ground Truth 湲곗?
				- ?뺤긽 ?곸옱: ?뚯깋 (?щ챸???덉쓬)
				- ??쟾 ?곸옱 (Ground Truth 湲곗?): 鍮④컙??(?좊챸??
				"""
				
				# 1. ?대깽??以묒씤 而⑦뀒?대꼫???먮옒 ???좎? (IN/OUT/UPDATE)
				if key in self.container_events:
					event_info = self.container_events[key]
					frames_elapsed = self.event_frame_counter - event_info['frame_count']
					if frames_elapsed < event_info['duration_frames']:
						event_type = event_info['event_type']
						if event_type == 'in': return self.in_color
						elif event_type == 'update': return self.update_color
						elif event_type == 'remove': return self.remove_color
						elif event_type == 'rehandling': return self.rehandling_color

				# 2. ??쟾 ?곸옱(Bad Stacking) ?먮떒 濡쒖쭅 (Ground Truth 湲곗?)
				if self.current_sim_time is not None and key in self.yard.container_positions:
					container_info = self.yard.containers.get(key)
					pos = self.yard.container_positions[key]
					block, x, y, z, yard_type = pos
					
					# 2痢??댁긽(z > 0)??寃쎌슦?먮쭔 ?꾨옒 而⑦뀒?대꼫? 鍮꾧탳
					if z > 0 and container_info:
						try:
							# [?섏젙?? 臾댁“嫄??ㅼ젣媛?CDT_true)??湲곗??쇰줈 ?됯?!  ?덉륫cdt硫?'std_CDT_pred'
							# ?곗씠?곗뿉 CDT_true媛 ?놁쑝硫?0?쇰줈 泥섎━ (鍮꾧탳 遺덇?)
							my_cdt = container_info.get('CDT_true', 0)
							
							# 諛붾줈 ?꾨옒 而⑦뀒?대꼫 李얘린
							yard_arr = self.yard.rf_yard if yard_type == 'RF' else self.yard.gp_yard
							below_key = yard_arr[block, x, y, z-1]
							
							# 40ft??寃쎌슦 ?꾨옒 吏吏? 泥섎━ (_2nd ?쒓굅)
							if below_key and isinstance(below_key, str):
								below_key = below_key.replace('_2nd', '')
							
							if below_key and below_key in self.yard.containers:
								below_info = self.yard.containers[below_key]
								# [?섏젙?? ?꾨옒 而⑦뀒?대꼫???ㅼ젣媛?CDT_true) 湲곗? ?덉륫cdt硫?'std_CDT_pred'
								below_cdt = below_info.get('CDT_true', 0)
								
								# [?먮떒 濡쒖쭅]
								# ?꾨옒 ?덈뒗 ??Below)????My)蹂대떎 CDT媛 ?묐떎 
								# = ?꾨옒 ?덈뒗 ?덉씠 ??鍮⑤━ ?섍????쒕떎
								# = ?섏쨷??由ы빖?ㅻ쭅 諛쒖깮 (BAD)
								if my_cdt is not None and below_cdt is not None:
									if float(below_cdt) < float(my_cdt): 
										return (1.0, 0.0, 0.0, 1.0) # 鍮④컙??(?뺤젙????쟾)

						except Exception:
							pass

				# 湲곕낯?? ?뚯깋 (?щ챸?꾨? ?쎄컙 二쇱뼱 鍮④컙?됱씠 ?뗫낫?닿쾶 ??
				return (0.8, 0.8, 0.8, 0.3)
	def _compute_actual_cdt_brightness(self, container_info):
		"""Return (brightness, current_time, in_time) derived from Actual CDT if available."""
		try:
			from datetime import datetime, timedelta
			if self.current_sim_time is None or not container_info:
				return (None, None, None)

			if isinstance(self.current_sim_time, str):
				current_time = datetime.fromisoformat(self.current_sim_time.replace('Z', '+00:00'))
			else:
				current_time = self.current_sim_time

			in_time = None
			if 'EVENT_TS' in container_info:
				val = container_info['EVENT_TS']
				if isinstance(val, str):
					in_time = datetime.fromisoformat(val.replace('Z', '+00:00'))
				else:
					in_time = val

			if current_time is None or in_time is None:
				return (None, current_time, in_time)

			actual_cdt_val = container_info.get('cdt_true') or container_info.get('CDT_true')
			# actual_cdt_val = container_info.get('std_CDT_pred') or container_info.get('no_std_CDT_pred') or container_info.get('CDT_true')
			if actual_cdt_val is None:
				return (None, current_time, in_time)

			actual_cdt_days = None
			if isinstance(actual_cdt_val, (int, float)):
				actual_cdt_days = max(0.0, float(actual_cdt_val) / 24.0)
			elif isinstance(actual_cdt_val, str):
				try:
					actual_cdt_days = max(0.0, float(actual_cdt_val) / 24.0)
				except ValueError:
					try:
						departure_time = datetime.fromisoformat(actual_cdt_val.replace('Z', '+00:00'))
						actual_cdt_days = max(0.0, (departure_time - in_time).total_seconds() / 86400.0)
					except Exception:
						pass
			elif isinstance(actual_cdt_val, datetime):
				actual_cdt_days = max(0.0, (actual_cdt_val - in_time).total_seconds() / 86400.0)
			elif isinstance(actual_cdt_val, timedelta):
				actual_cdt_days = max(0.0, actual_cdt_val.total_seconds() / 86400.0)

			if actual_cdt_days is None:
				return (None, current_time, in_time)

            
			if actual_cdt_days <= 0.25:  # 6 hours
				brightness = 0.95
			elif actual_cdt_days <= 0.5:  # 12 hours
				brightness = 0.75
			elif actual_cdt_days <= 1:    # 1 day
				brightness = 0.55
			elif actual_cdt_days <= 3:    # 3 days
				brightness = 0.40
			else:
				brightness = 0.25

			return (brightness, current_time, in_time)
		except Exception:
			return (None, None, None)

	def _scale_color(self, color, brightness):
		"""Apply brightness scaling to an RGBA tuple."""
		r = max(0.0, min(1.0, color[0] * brightness))
		g = max(0.0, min(1.0, color[1] * brightness))
		b = max(0.0, min(1.0, color[2] * brightness))
		alpha = color[3] if len(color) > 3 else 1.0
		return (r, g, b, alpha)
	
	def _create_box(self, pos: Tuple[float, float, float], size: Tuple[float, float, float], 
					color: Tuple[float, float, float, float], parent: NodePath) -> NodePath:
		"""Create a 3D box (container) at the given position with specified size and color."""
		from panda3d.core import CollisionBox, CollisionNode
		
		# Create geometry for a box
		format = GeomVertexFormat.getV3n3c4()
		vdata = GeomVertexData("box", format, Geom.UHStatic)
		vdata.setNumRows(24)  # 6 faces * 4 vertices
		
		vertex = GeomVertexWriter(vdata, "vertex")
		normal = GeomVertexWriter(vdata, "normal")
		vcolor = GeomVertexWriter(vdata, "color")
		
		x, y, z = pos
		sx, sy, sz = size
		
		# Define 8 corners of the box
		v = [
			(x, y, z), (x + sx, y, z), (x + sx, y + sy, z), (x, y + sy, z),  # bottom
			(x, y, z + sz), (x + sx, y, z + sz), (x + sx, y + sy, z + sz), (x, y + sy, z + sz)  # top
		]
		
		# Define 6 faces with correct winding order (counter-clockwise when viewed from outside)
		# Each face: (vertices in CCW order, normal pointing outward)
		faces = [
			# bottom (z=0) - looking from below, CCW
			((0, 3, 2, 1), (0, 0, -1)),
			# top (z=sz) - looking from above, CCW
			((4, 5, 6, 7), (0, 0, 1)),
			# front (y=0) - looking from front, CCW
			((0, 1, 5, 4), (0, -1, 0)),
			# back (y=sy) - looking from back, CCW
			((2, 3, 7, 6), (0, 1, 0)),
			# left (x=0) - looking from left, CCW
			((0, 4, 7, 3), (-1, 0, 0)),
			# right (x=sx) - looking from right, CCW
			((1, 2, 6, 5), (1, 0, 0)),
		]
		
		triangles = GeomTriangles(Geom.UHStatic)
		vertex_index = 0
		
		for face_indices, face_normal in faces:
			i0, i1, i2, i3 = face_indices
			# Add 4 vertices for this face
			for vi in [i0, i1, i2, i3]:
				vertex.addData3(*v[vi])
				normal.addData3(*face_normal)
				vcolor.addData4(*color)
			
			# Add 2 triangles (quad) with correct winding order
			triangles.addVertices(vertex_index, vertex_index + 1, vertex_index + 2)
			triangles.addVertices(vertex_index, vertex_index + 2, vertex_index + 3)
			vertex_index += 4
		
		geom = Geom(vdata)
		geom.addPrimitive(triangles)
		
		node = GeomNode("box")
		node.addGeom(geom)
		
		box_np = parent.attachNewNode(node)
		# Disable backface culling to ensure all faces are visible from any angle
		box_np.setTwoSided(True)
		
		# Add collision solid for picking
		# Create a collision box centered at the box position
		center_x = x + sx / 2
		center_y = y + sy / 2
		center_z = z + sz / 2
		collision_solid = CollisionBox((center_x, center_y, center_z), sx / 2, sy / 2, sz / 2)
		
		collision_node = CollisionNode('container_collision')
		collision_node.addSolid(collision_solid)
		collision_node.setIntoCollideMask(1)  # Set collision mask
		collision_np = box_np.attachNewNode(collision_node)
		
		return box_np
	
	def _create_grid(self, parent: NodePath, block_idx: int, is_rf: bool = False):
		"""Create ground grid for a block."""
		w, h = self.cfg.width, self.cfg.height
		
		# Layout blocks in rows of 10
		BLOCKS_PER_ROW = 10
		row = block_idx // BLOCKS_PER_ROW
		col = block_idx % BLOCKS_PER_ROW
		
		offset_x = col * (w + 5)  # Horizontal spacing
		# Each row needs space for both GP and RF yards
		offset_y_base = row * ((h + 5) * 2 + 10)  # 2 yards (GP+RF) + gap between rows
		offset_y = offset_y_base + ((h + 5) if is_rf else 0)  # RF is behind GP
		
		format = GeomVertexFormat.getV3c4()
		vdata = GeomVertexData("grid", format, Geom.UHStatic)
		
		vertex = GeomVertexWriter(vdata, "vertex")
		color = GeomVertexWriter(vdata, "color")
		
		lines = GeomLines(Geom.UHStatic)
		v_idx = 0
		
		# X-direction lines
		for x in range(w + 1):
			vertex.addData3(offset_x + x, offset_y, 0)
			color.addData4(*self.grid_color)
			vertex.addData3(offset_x + x, offset_y + h, 0)
			color.addData4(*self.grid_color)
			lines.addVertices(v_idx, v_idx + 1)
			v_idx += 2
		
		# Y-direction lines (with lane markers)
		for y in range(h + 1):
			line_color = self.lane_color if (self.lane_every and y % self.lane_every == 0 and y != 0 and y != h) else self.grid_color
			vertex.addData3(offset_x, offset_y + y, 0)
			color.addData4(*line_color)
			vertex.addData3(offset_x + w, offset_y + y, 0)
			color.addData4(*line_color)
			lines.addVertices(v_idx, v_idx + 1)
			v_idx += 2
		
		geom = Geom(vdata)
		geom.addPrimitive(lines)
		
		node = GeomNode("grid")
		node.addGeom(geom)
		
		grid_np = parent.attachNewNode(node)
		
		# Add yard type label (GP or RF)
		#self._create_yard_label(parent, block_idx, is_rf)
		
		return grid_np
	
	def _create_yard_label(self, parent: NodePath, block_idx: int, is_rf: bool):
		"""Create a 3D text label to identify yard type (GP or RF)."""
		from panda3d.core import TextNode
		
		w, h = self.cfg.width, self.cfg.height
		
		# Layout blocks in rows of 10
		BLOCKS_PER_ROW = 10
		row = block_idx // BLOCKS_PER_ROW
		col = block_idx % BLOCKS_PER_ROW
		
		offset_x = col * (w + 5)
		# Each row needs space for both GP and RF yards
		offset_y_base = row * ((h + 5) * 2 + 10)
		offset_y = offset_y_base + ((h + 5) if is_rf else 0)
		
		# Create text node
		text_node = TextNode(f'yard_label_{block_idx}_{"RF" if is_rf else "GP"}')
		yard_type = f"RF{block_idx}" if is_rf else f"GP{block_idx}"
		text_node.setText(yard_type)
		
		# Set text properties
		if is_rf:
			text_node.setTextColor(0.9, 0.5, 0.3, 1.0)  # Orange for RF
		else:
			text_node.setTextColor(0.3, 0.7, 0.9, 1.0)  # Blue for GP
		
		text_node.setAlign(TextNode.ACenter)
		
		# Attach to yard_labels_root (so it can be toggled with 't' key)
		text_np = self.yard_labels_root.attachNewNode(text_node)
		
		# Position at the front center of the yard, slightly elevated
		center_x = offset_x + w / 2
		center_y = offset_y - 2  # In front of the yard
		text_np.setPos(center_x, center_y, 0.5)
		
		# Scale and rotate to face camera
		text_np.setScale(2.5)  # Large text
		text_np.setBillboardPointEye()  # Always face the camera
		
		return text_np
	
	def _create_ocean(self, layout_width, layout_height):
		"""Create ocean/sea water behind the port."""
		from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
		from panda3d.core import Geom, GeomTriangles, GeomNode
		
		format = GeomVertexFormat.getV3c4()
		vdata = GeomVertexData("ocean", format, Geom.UHStatic)
		vertex = GeomVertexWriter(vdata, "vertex")
		color = GeomVertexWriter(vdata, "color")
		
		# Ocean extends beyond the front of the yard
		ocean_depth = 80
		x1, y1 = -50, -80
		x2, y2 = layout_width + 50, -10
		
		# Ocean color (dark blue)
		ocean_color = (0.1, 0.3, 0.5, 1.0)
		
		# Four corners
		vertex.addData3(x1, y1, -0.5)
		color.addData4(*ocean_color)
		
		vertex.addData3(x2, y1, -0.5)
		color.addData4(*ocean_color)
		
		vertex.addData3(x2, y2, -0.5)
		color.addData4(*ocean_color)
		
		vertex.addData3(x1, y2, -0.5)
		color.addData4(*ocean_color)
		
		# Create triangles
		tris = GeomTriangles(Geom.UHStatic)
		tris.addVertices(0, 1, 2)
		tris.addVertices(0, 2, 3)
		
		geom = Geom(vdata)
		geom.addPrimitive(tris)
		
		node = GeomNode("ocean")
		node.addGeom(geom)
		
		ocean_np = self.port_root.attachNewNode(node)
		return ocean_np
	
	def _add_port_decorations(self):
		"""Add realistic port decorations: cranes, roads, fences, lighting, buildings."""
		from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
		from panda3d.core import Geom, GeomTriangles, GeomNode
		
		# Calculate layout dimensions
		BLOCKS_PER_ROW = 10
		total_rows = (self.total_blocks + BLOCKS_PER_ROW - 1) // BLOCKS_PER_ROW
		w, h = self.cfg.width, self.cfg.height
		layout_width = min(self.total_blocks, BLOCKS_PER_ROW) * (w + 5)
		layout_height = total_rows * ((h + 5) * 2 + 10)
		
		# 0. Ocean (behind everything)
		self._create_ocean(layout_width, layout_height)
		
		# 1. Ground/Pavement (asphalt look)
		self._create_ground(layout_width, layout_height)
		
		# 2. Roads between blocks
		self._create_roads(layout_width, layout_height)
		
		# 3. Ships at the quay - DISABLED
		# self._create_ships(layout_width, layout_height)
		
		# 4. Quay Cranes (QC) near the water - DISABLED
		# self._create_quay_cranes(layout_width, layout_height)
		
		# 5. Perimeter fence
		self._create_perimeter_fence(layout_width, layout_height)
		
		# 7. Lighting poles
		self._create_lighting_poles(layout_width, layout_height)
		
		# 8. Office building
		self._create_office_building(layout_width, layout_height)
		
		# 9. Gate entrance
		self._create_gate_entrance(layout_width, layout_height)
	
	def _create_ground(self, layout_width, layout_height):
		"""Create ground/pavement."""
		from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
		from panda3d.core import Geom, GeomTriangles, GeomNode
		
		format = GeomVertexFormat.getV3c4()
		vdata = GeomVertexData("ground", format, Geom.UHStatic)
		vertex = GeomVertexWriter(vdata, "vertex")
		color = GeomVertexWriter(vdata, "color")
		
		# Expand ground beyond the yard (larger margin for wider ground)
		margin = 1000
		x1, y1 = -margin, -margin
		x2, y2 = layout_width + margin, layout_height + margin
		
		# Ground color (dark gray asphalt)
		ground_color = (0.3, 0.3, 0.35, 1.0)
		
		# Four corners
		vertex.addData3(x1, y1, -0.1)
		color.addData4(*ground_color)
		
		vertex.addData3(x2, y1, -0.1)
		color.addData4(*ground_color)
		
		vertex.addData3(x2, y2, -0.1)
		color.addData4(*ground_color)
		
		vertex.addData3(x1, y2, -0.1)
		color.addData4(*ground_color)
		
		# Create triangles
		tris = GeomTriangles(Geom.UHStatic)
		tris.addVertices(0, 1, 2)
		tris.addVertices(0, 2, 3)
		
		geom = Geom(vdata)
		geom.addPrimitive(tris)
		
		node = GeomNode("ground")
		node.addGeom(geom)
		
		ground_np = self.yard_root.attachNewNode(node)
		return ground_np
	
	def _create_roads(self, layout_width, layout_height):
		"""Create roads between yard blocks."""
		from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
		from panda3d.core import Geom, GeomTriangles, GeomNode
		
		BLOCKS_PER_ROW = 10
		total_rows = (self.total_blocks + BLOCKS_PER_ROW - 1) // BLOCKS_PER_ROW
		w, h = self.cfg.width, self.cfg.height
		
		road_color = (0.25, 0.25, 0.28, 1.0)  # Slightly lighter than ground
		road_width = 2.5  # Slightly narrower to avoid overlap
		
		# Horizontal roads between rows (between GP-RF pairs)
		for row in range(total_rows):
			# Road between GP and RF within the same row
			y_pos = row * ((h + 5) * 2 + 10) + h + 2.5 - road_width / 2
			self._create_road_segment(0, y_pos, layout_width, road_width, road_color)
			
			# Road after each row pair (if not the last row)
			if row < total_rows - 1:
				y_pos = (row + 1) * ((h + 5) * 2 + 10) - 5 - road_width / 2
				self._create_road_segment(0, y_pos, layout_width, road_width, road_color)
		
		# Vertical roads between columns
		for col in range(1, min(self.total_blocks, BLOCKS_PER_ROW)):
			x_pos = col * (w + 5) - 2.5 - road_width / 2
			self._create_road_segment(x_pos, 0, road_width, layout_height, road_color, vertical=True)
	
	def _create_road_segment(self, x, y, width, height, color, vertical=False):
		"""Create a single road segment."""
		from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
		from panda3d.core import Geom, GeomTriangles, GeomNode
		
		format = GeomVertexFormat.getV3c4()
		vdata = GeomVertexData("road", format, Geom.UHStatic)
		vertex = GeomVertexWriter(vdata, "vertex")
		vcolor = GeomVertexWriter(vdata, "color")
		
		# Create rectangle
		vertex.addData3(x, y, -0.05)
		vcolor.addData4(*color)
		
		vertex.addData3(x + width, y, -0.05)
		vcolor.addData4(*color)
		
		vertex.addData3(x + width, y + height, -0.05)
		vcolor.addData4(*color)
		
		vertex.addData3(x, y + height, -0.05)
		vcolor.addData4(*color)
		
		tris = GeomTriangles(Geom.UHStatic)
		tris.addVertices(0, 1, 2)
		tris.addVertices(0, 2, 3)
		
		geom = Geom(vdata)
		geom.addPrimitive(tris)
		
		node = GeomNode("road_segment")
		node.addGeom(geom)
		
		road_np = self.yard_root.attachNewNode(node)
		return road_np
	
	def _create_ships(self, layout_width, layout_height):
		"""Create a single container ship docked at the quay."""
		# Create 1 ship at the center, closer to the port
		ship_x = layout_width / 2
		ship_y = -35  # Moved closer to port (was -55)
		ship_z = 0
		
		self._create_single_ship(ship_x, ship_y, ship_z)
	
	def _create_single_ship(self, x, y, z):
		"""Create a single massive container ship with detailed features (realistic scale)."""
		# All ship elements attached to port_root (not yard_root)
		parent = self.port_root
		
		# Ship hull (dark blue-gray with bow and stern shape)
		hull_color = (0.2, 0.25, 0.35, 1.0)
		deck_color = (0.3, 0.3, 0.3, 1.0)
		rust_color = (0.4, 0.3, 0.25, 1.0)
		
		# Main hull (much larger - typical container ship is 300-400m long)
		# Scale: Each container is ~2.4m high, 6m long, 2.4m wide
		# Making ship roughly 100m long in our scale
		ship_length = 100
		ship_width = 22
		ship_height = 12
		
		self._create_box_at(x, y, z - 5, ship_length, ship_width, ship_height, hull_color, "ship_hull", parent)
		
		# Bow (front of ship, tapered)
		self._create_box_at(x + ship_length/2 + 3, y, z - 4, 6, ship_width - 4, ship_height - 2, hull_color, "ship_bow", parent)
		self._create_box_at(x + ship_length/2 + 8, y, z - 3, 4, ship_width - 8, ship_height - 4, hull_color, "ship_bow_tip", parent)
		
		# Stern (back of ship)
		self._create_box_at(x - ship_length/2 - 3, y, z - 4, 6, ship_width - 2, ship_height - 2, hull_color, "ship_stern", parent)
		
		# Waterline stripe (red/rust color) - scaled proportionally
		self._create_box_at(x, y, z - 0.5, ship_length + 4, ship_width + 0.5, 0.3, rust_color, "ship_waterline", parent)
		
		# Deck - scaled to match ship dimensions
		self._create_box_at(x, y, z + ship_height/3, ship_length - 4, ship_width, 0.5, deck_color, "ship_deck", parent)
		
		# Bridge/Superstructure (at the back, more detailed) - scaled
		bridge_x = x - ship_length * 0.4  # Position at stern
		self._create_box_at(bridge_x, y, z + ship_height/3 + 0.5, ship_length * 0.17, ship_width * 0.55, ship_height, (0.9, 0.9, 0.9, 1.0), "ship_bridge", parent)
		
		# Bridge levels (multiple decks) - scaled proportionally
		self._create_box_at(bridge_x, y, z + ship_height/3 + ship_height + 0.5, ship_length * 0.15, ship_width * 0.45, ship_height * 0.25, (0.85, 0.85, 0.85, 1.0), "ship_bridge_upper", parent)
		self._create_box_at(bridge_x, y, z + ship_height/3 + ship_height * 1.25 + 0.5, ship_length * 0.12, ship_width * 0.36, ship_height * 0.17, (0.8, 0.8, 0.8, 1.0), "ship_wheelhouse", parent)
		
		# Smokestack (taller and more prominent) - scaled
		self._create_box_at(bridge_x, y, z + ship_height/3 + ship_height * 1.42 + 0.5, ship_length * 0.05, ship_width * 0.14, ship_height * 0.58, (0.8, 0.2, 0.2, 1.0), "ship_stack", parent)
		
		# Stack cap - scaled
		self._create_box_at(bridge_x, y, z + ship_height/3 + ship_height * 2.0 + 0.5, ship_length * 0.058, ship_width * 0.16, ship_height * 0.04, (0.9, 0.3, 0.3, 1.0), "ship_stack_cap", parent)
		
		# Mast/Radar - scaled
		self._create_box_at(bridge_x, y, z + ship_height/3 + ship_height * 2.04 + 0.5, ship_length * 0.008, ship_width * 0.023, ship_height * 0.33, (0.5, 0.5, 0.5, 1.0), "ship_mast", parent)
		
		# Deck cranes (ship's own cranes) - scaled and repositioned
		self._create_box_at(x - ship_length * 0.17, y - ship_width * 0.23, z + ship_height/3 + 0.5, ship_length * 0.025, ship_width * 0.068, ship_height * 0.67, (0.9, 0.7, 0.2, 1.0), "ship_crane1", parent)
		self._create_box_at(x + ship_length * 0.083, y + ship_width * 0.23, z + ship_height/3 + 0.5, ship_length * 0.025, ship_width * 0.068, ship_height * 0.67, (0.9, 0.7, 0.2, 1.0), "ship_crane2", parent)
		self._create_box_at(x + ship_length * 0.083, y + ship_width * 0.23, z + ship_height/3 + 0.5, ship_length * 0.025, ship_width * 0.068, ship_height * 0.67, (0.9, 0.7, 0.2, 1.0), "ship_crane2")
		
		# More container stacks on deck (colorful and detailed)
		container_colors = [
			(0.8, 0.2, 0.2, 1.0),  # Red
			(0.2, 0.6, 0.8, 1.0),  # Blue
			(0.2, 0.8, 0.3, 1.0),  # Green
			(0.9, 0.7, 0.2, 1.0),  # Yellow
			(0.7, 0.3, 0.7, 1.0),  # Purple
			(1.0, 0.5, 0.0, 1.0),  # Orange
			(0.3, 0.3, 0.3, 1.0),  # Dark gray
			(0.6, 0.6, 0.6, 1.0),  # Light gray
		]
		
		# Container stacks using same unit size as yard containers
		# Yard container unit: 1x1x1, so ship containers should match
		container_unit = 1.0  # Same as yard container base unit
		
		# Fill the ship deck with containers - VERY dense packing (tightly packed)
		num_bays_length = 35  # Many more bays along the ship length
		num_rows_width = 10   # Many more rows across the width
		bay_start = -ship_length * 0.45
		bay_end = ship_length * 0.4
		bay_spacing = (bay_end - bay_start) / num_bays_length
		
		# Width positions for container rows - very tight spacing
		width_start = -ship_width * 0.42
		width_spacing = (ship_width * 0.84) / num_rows_width
		
		for i in range(num_bays_length):
			cx = bay_start + i * bay_spacing
			
			for w_idx in range(num_rows_width):
				cy = width_start + w_idx * width_spacing
				color = container_colors[(i + w_idx) % len(container_colors)]
				
				# High stacks (6-8 layers) - very full ship
				layers = 6 + (i % 3)
				
				for layer in range(layers):
					self._create_box_at(x + cx, y + cy, z + ship_height/3 + 1.0 + layer * container_unit, 
						container_unit, container_unit, container_unit, color, "ship_container", parent)
		
		# Anchor (visible at bow) - scaled
		self._create_box_at(x + ship_length * 0.47, y - ship_width * 0.32, z, ship_length * 0.017, ship_width * 0.045, ship_height * 0.17, (0.3, 0.3, 0.3, 1.0), "ship_anchor", parent)
	
	def _create_quay_cranes(self, layout_width, layout_height):
		"""Create Quay Cranes (Ship-to-Shore cranes) at the waterfront."""
		# Place 4 QCs on the land side (between ship and yard)
		# Ship is at y=-35, yard is at y=0 (but moved +40 by yard_root)
		ship_x = layout_width / 2
		qc_y = 5  # Moved closer to yard, further from ship (was -10)
		
		# QCs positioned along the ship width, on the land
		qc_spacing = 15  # Spacing between QCs along the ship
		qc_positions = [
			(ship_x - qc_spacing * 1.5, qc_y, 0),  # QC 1 (left)
			(ship_x - qc_spacing * 0.5, qc_y, 0),  # QC 2
			(ship_x + qc_spacing * 0.5, qc_y, 0),  # QC 3
			(ship_x + qc_spacing * 1.5, qc_y, 0),  # QC 4 (right)
		]
		
		for x, y, z in qc_positions:
			self._create_single_quay_crane(x, y, z)
	
	def _create_single_quay_crane(self, x, y, z):
		"""Create a single realistic Quay Crane (STS crane) - smaller size, blue color."""
		crane_color = (0.2, 0.5, 1.0, 1.0)  # Blue color
		steel_color = (0.15, 0.4, 0.8, 1.0)  # Darker blue for structure
		
		# Reduced size - smaller legs and height
		leg_width = 0.8  # Thinner
		leg_depth = 0.8
		leg_height = 25  # Lower height
		
		# 4 main legs in portal frame configuration
		# Waterside legs (facing the ships) - positioned to not overlap with ship
		# Ship is at y=-35 with length 100m (roughly y=-85 to y=15)
		# Place waterside legs at y=-18 to avoid overlap with ship front (y=15)
		self._create_box_at(x - 6, y - 23, z, leg_width, leg_depth, leg_height, crane_color, "qc_leg_ws_left")
		self._create_box_at(x + 6, y - 23, z, leg_width, leg_depth, leg_height, crane_color, "qc_leg_ws_right")
		
		# Landside legs (facing the yard) - closer to the yard
		self._create_box_at(x - 6, y + 15, z, leg_width, leg_depth, leg_height, crane_color, "qc_leg_ls_left")
		self._create_box_at(x + 6, y + 15, z, leg_width, leg_depth, leg_height, crane_color, "qc_leg_ls_right")
		
		# Cross bracing on legs (for structural realism)
		# Waterside bracing
		self._create_box_at(x, y - 23, z + leg_height * 0.33, 12, 0.3, 0.3, steel_color, "qc_brace_ws1")
		self._create_box_at(x, y - 23, z + leg_height * 0.67, 12, 0.3, 0.3, steel_color, "qc_brace_ws2")
		
		# Landside bracing
		self._create_box_at(x, y + 15, z + leg_height * 0.33, 12, 0.3, 0.3, steel_color, "qc_brace_ls1")
		self._create_box_at(x, y + 15, z + leg_height * 0.67, 12, 0.3, 0.3, steel_color, "qc_brace_ls2")
		
		# Top portal beam (connects all 4 legs at top) - spans from water to land (longer)
		self._create_box_at(x, y - 4, z + leg_height, 14, 40, 1.0, crane_color, "qc_portal_beam")
		
		# Boom (extends over water toward the ship) - longer boom to reach ship
		boom_length = 38
		boom_height = 1.2
		self._create_box_at(x, y - 30, z + leg_height + 3, 1.2, boom_length, boom_height, crane_color, "qc_boom_main")
		
		# Boom support cables (A-frame style)
		self._create_box_at(x - 3, y - 4, z + leg_height + 5, 0.3, 20, 0.3, steel_color, "qc_boom_cable1")
		self._create_box_at(x + 3, y - 4, z + leg_height + 5, 0.3, 20, 0.3, steel_color, "qc_boom_cable2")
		
		# Trolley on boom - positioned over the ship
		trolley_y = y - 28
		self._create_box_at(x, trolley_y, z + leg_height + 2, 2.2, 2, 1.5, (0.7, 0.7, 0.7, 1.0), "qc_trolley")
		
		# Spreader hanging from trolley
		self._create_box_at(x, trolley_y, z + 12, 4, 2, 0.4, (0.85, 0.85, 0.3, 1.0), "qc_spreader")
		
		# Cables from trolley to spreader
		self._create_box_at(x - 1.5, trolley_y, z + 16, 0.15, 0.15, 8, (0.2, 0.2, 0.2, 1.0), "qc_cable1")
		self._create_box_at(x + 1.5, trolley_y, z + 16, 0.15, 0.15, 8, (0.2, 0.2, 0.2, 1.0), "qc_cable2")
		self._create_box_at(x - 1.5, trolley_y + 0.8, z + 16, 0.15, 0.15, 8, (0.2, 0.2, 0.2, 1.0), "qc_cable3")
		self._create_box_at(x + 1.5, trolley_y + 0.8, z + 16, 0.15, 0.15, 8, (0.2, 0.2, 0.2, 1.0), "qc_cable4")
		
		# Back stay (support structure) - adjusted for landside
		self._create_box_at(x, y + 15, z + leg_height + 2, 1.0, 4, 1.2, crane_color, "qc_backstay")
		
		# Machinery house (on the portal beam) - centered
		self._create_box_at(x, y - 4, z + leg_height + 1.0, 5, 5, 2, (0.75, 0.75, 0.75, 1.0), "qc_machinery")
		
		# Operator cabin (high up on the landside) - adjusted
		self._create_box_at(x + 5, y + 12, z + leg_height - 5, 1.8, 2, 2, crane_color, "qc_cabin")
		
		# Cabin windows (detail)
		self._create_box_at(x + 5.4, y + 12, z + leg_height - 4.5, 0.08, 1.6, 1.2, (0.6, 0.8, 1.0, 1.0), "qc_cabin_window")
		
		# Navigation lights - smaller, on waterside legs
		self._create_box_at(x - 6, y - 23, z + leg_height, 0.3, 0.3, 0.3, (1.0, 0.0, 0.0, 1.0), "qc_light_red")
		self._create_box_at(x + 6, y - 23, z + leg_height, 0.3, 0.3, 0.3, (0.0, 1.0, 0.0, 1.0), "qc_light_green")
	
	def _create_perimeter_fence(self, layout_width, layout_height):
		"""Create perimeter fence around the yard."""
		fence_color = (0.5, 0.5, 0.55, 1.0)  # Gray
		fence_height = 3
		post_spacing = 5
		
		# Calculate actual yard boundaries (blocks start at 0, end at layout size)
		margin = 10  # Distance from yard edge to fence
		
		# Front fence (south) - in front of the yard
		for x in range(0, int(layout_width) + post_spacing, post_spacing):
			self._create_box_at(x, -margin, 0, 0.3, 0.3, fence_height, fence_color, "fence_post", self.decorations_root)
		
		# Back fence (north) - behind the yard
		for x in range(0, int(layout_width) + post_spacing, post_spacing):
			self._create_box_at(x, layout_height + margin, 0, 0.3, 0.3, fence_height, fence_color, "fence_post", self.decorations_root)
		
		# Left fence (west) - left side of the yard
		for y in range(-margin, int(layout_height) + margin + post_spacing, post_spacing):
			self._create_box_at(-margin, y, 0, 0.3, 0.3, fence_height, fence_color, "fence_post", self.decorations_root)
		
		# Right fence (east) - right side of the yard
		for y in range(-margin, int(layout_height) + margin + post_spacing, post_spacing):
			self._create_box_at(layout_width + margin, y, 0, 0.3, 0.3, fence_height, fence_color, "fence_post", self.decorations_root)
	
	def _create_lighting_poles(self, layout_width, layout_height):
		"""Create lighting poles throughout the yard."""
		pole_color = (0.4, 0.4, 0.4, 1.0)  # Dark gray
		light_color = (1.0, 1.0, 0.9, 1.0)  # Warm white
		pole_height = 15
		
		BLOCKS_PER_ROW = 10
		w, h = self.cfg.width, self.cfg.height
		total_rows = (self.total_blocks + BLOCKS_PER_ROW - 1) // BLOCKS_PER_ROW
		
		# Place poles at regular intervals within the actual yard boundaries
		# Spacing: every 3 columns and between each row pair
		for col in range(0, min(self.total_blocks, BLOCKS_PER_ROW), 3):
			for row in range(total_rows):
				x = col * (w + 5) + w / 2
				# Position between GP and RF yards in each row
				y = row * ((h + 5) * 2 + 10) + h + 2.5
				
				# Pole
				self._create_box_at(x, y, 0, 0.4, 0.4, pole_height, pole_color, "light_pole", self.decorations_root)
				
				# Light fixture on top
				self._create_box_at(x, y, pole_height, 1.2, 1.2, 0.8, light_color, "light_fixture", self.decorations_root)
	
	def _create_office_building(self, layout_width, layout_height):
		"""Create a small office/control building."""
		building_color = (0.85, 0.85, 0.9, 1.0)  # Light gray
		roof_color = (0.6, 0.3, 0.2, 1.0)  # Brown
		
		# Position at back corner (inland side), outside the fence boundary
		x, y = -20, layout_height + 20  # Moved to back/inland side
		
		# Main building
		self._create_box_at(x, y, 0, 8, 6, 5, building_color, "office_building", self.decorations_root)
		
		# Roof
		self._create_box_at(x, y, 5, 8.5, 6.5, 0.5, roof_color, "office_roof", self.decorations_root)
		
		# Windows (small dark squares)
		window_color = (0.2, 0.3, 0.4, 1.0)  # Dark blue
		for wx in [x - 2, x + 2]:
			for wz in [2, 3.5]:
				self._create_box_at(wx, y + 3.1, wz, 0.8, 0.1, 0.8, window_color, "window", self.decorations_root)
	
	def _create_gate_entrance(self, layout_width, layout_height):
		"""Create entrance gate."""
		gate_color = (0.7, 0.7, 0.2, 1.0)  # Yellow
		
		# Gate posts at the back (inland side), centered
		x, y = layout_width / 2 - 5, layout_height + 15  # Moved to back/inland side
		self._create_box_at(x, y, 0, 0.5, 0.5, 4, gate_color, "gate_post", self.decorations_root)
		self._create_box_at(x + 10, y, 0, 0.5, 0.5, 4, gate_color, "gate_post", self.decorations_root)
		
		# Gate bar (horizontal)
		self._create_box_at(x + 5, y, 3.5, 10, 0.3, 0.3, gate_color, "gate_bar", self.decorations_root)
	
	def _create_box_at(self, x, y, z, width, depth, height, color, name="decoration", parent=None):
		"""Helper function to create a colored box at specific position."""
		from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
		from panda3d.core import Geom, GeomTriangles, GeomNode
		
		if parent is None:
			parent = self.yard_root
		
		format = GeomVertexFormat.getV3c4()
		vdata = GeomVertexData(name, format, Geom.UHStatic)
		vertex = GeomVertexWriter(vdata, "vertex")
		vcolor = GeomVertexWriter(vdata, "color")
		
		# 8 corners of the box
		hw, hd, hh = width / 2, depth / 2, height / 2
		corners = [
			(x - hw, y - hd, z),
			(x + hw, y - hd, z),
			(x + hw, y + hd, z),
			(x - hw, y + hd, z),
			(x - hw, y - hd, z + height),
			(x + hw, y - hd, z + height),
			(x + hw, y + hd, z + height),
			(x - hw, y + hd, z + height),
		]
		
		for corner in corners:
			vertex.addData3(*corner)
			vcolor.addData4(*color)
		
		# 6 faces
		faces = [
			[0, 1, 2, 3],  # bottom
			[4, 5, 6, 7],  # top
			[0, 1, 5, 4],  # front
			[2, 3, 7, 6],  # back
			[0, 3, 7, 4],  # left
			[1, 2, 6, 5],  # right
		]
		
		tris = GeomTriangles(Geom.UHStatic)
		for face in faces:
			tris.addVertices(face[0], face[1], face[2])
			tris.addVertices(face[0], face[2], face[3])
		
		geom = Geom(vdata)
		geom.addPrimitive(tris)
		
		node = GeomNode(name)
		node.addGeom(geom)
		
		decoration_np = parent.attachNewNode(node)
		return decoration_np
	
	def _build_scene(self):
		"""Build the complete 3D scene with all containers and grids."""
		# Clear existing yard
		self.yard_root.removeNode()
		self.yard_root = self.render.attachNewNode("yard_root")
		
		# Clear and recreate port_root (for ocean)
		self.port_root.removeNode()
		self.port_root = self.render.attachNewNode("port_root")
		
		# Clear and recreate decorations_root (for fence, lighting, office, gate)
		self.decorations_root.removeNode()
		self.decorations_root = self.render.attachNewNode("decorations_root")
		
		# Clear and recreate yard_labels_root (for GP0, RF0, etc.)
		self.yard_labels_root.removeNode()
		self.yard_labels_root = self.render.attachNewNode("yard_labels_root")
		
		# Add port decorations first (so they appear behind containers)
		self._add_port_decorations()
		
		# Build each block (GP and RF)
		for block_idx in range(self.total_blocks):
			# Create grids and containers for GP yard
			if block_idx < self.gp_blocks:
				self._create_grid(self.yard_root, block_idx, is_rf=False)
				self._build_containers_for_block(self.yard.gp_yard, block_idx, is_rf=False)
			
			# Create grids and containers for RF yard
			if block_idx < self.rf_blocks:
				self._create_grid(self.yard_root, block_idx, is_rf=True)
				self._build_containers_for_block(self.yard.rf_yard, block_idx, is_rf=True)
		
		# Apply current visibility state after rebuilding scene
		self._apply_decorations_visibility()
	
	def _build_containers_for_block(self, yard_array, block_idx: int, is_rf: bool = False):
		"""Build container geometry for one block."""
		w, h, d = self.cfg.width, self.cfg.height, self.cfg.depth
		
		# Layout blocks in rows of 10
		BLOCKS_PER_ROW = 10
		row = block_idx // BLOCKS_PER_ROW
		col = block_idx % BLOCKS_PER_ROW
		
		offset_x = col * (w + 5)
		# Each row needs space for both GP and RF yards
		offset_y_base = row * ((h + 5) * 2 + 10)
		offset_y = offset_y_base + ((h + 5) if is_rf else 0)
		
		processed = set()
		
		for x in range(w):
			for y in range(h):
				for z in range(d):
					key = yard_array[block_idx, x, y, z]
					if key is None or key in processed:
						continue
					if isinstance(key, str) and key.endswith('_2nd'):
						continue
					
					# Get container info
					info = self.yard.containers.get(key)
					base_col = self.rf_base if is_rf else self.gp_base
					color = self._color_for_key(key, base_col)
					
					# Determine size (1 for 20ft, 2 for 40ft)
					span = 1
					if info is not None:
						try:
							# Try CARGO_SIZE first, then SZTP2
							size_val = info.get('CARGO_SIZE', info.get('SZTP2'))
							span = 2 if self.yard.get_container_size(size_val) == '40ft' else 1
						except Exception:
							span = 1
					
					# Create box (40ft containers span 2 units in X)
					box_size = (span * 0.95, 0.95, 0.95)  # Slightly smaller for visual separation
					box_pos = (offset_x + x, offset_y + y, z)
					
					box_np = self._create_box(box_pos, box_size, color, self.yard_root)
					
					# Track the NodePath for this container
					self.container_nodes[key] = box_np
					
					processed.add(key)
	
	def refresh_scene(self):
		"""Rebuild the entire scene (call this after simulation updates)."""
		# Clear container node tracking
		self.container_nodes.clear()
		self._build_scene()
	
	def update_container_colors(self):
		"""Update colors of existing containers based on dwell time without rebuilding the scene."""
		from panda3d.core import GeomNode, GeomVertexWriter
		
		colors_updated = 0
		sample_logs = []  # For debugging - log a few samples
		
		for key, node_path in list(self.container_nodes.items()):
			# Skip animating containers
			if key.startswith('_animating_'):
				continue
			
			# Get container info and calculate color
			container_info = self.yard.containers.get(key)
			if not container_info:
				continue
			
			# Determine if it's RF or GP (from the node's position or stored info)
			# For now, we'll use a fallback base color
			base_col = self.gp_base  # Default to GP color
			color = self._color_for_key(key, base_col)
			
			# Log sample for first 3 containers (for debugging)
			if len(sample_logs) < 3 and 'no_std_CDT_pred' in container_info:
				try:
					from datetime import datetime, timedelta
					predicted_cdt_hours = float(container_info['no_std_CDT_pred'])
					if isinstance(self.current_sim_time, str):
						current_time = datetime.fromisoformat(self.current_sim_time.replace('Z', '+00:00'))
					else:
						current_time = self.current_sim_time
					in_time = datetime.fromisoformat(container_info['EVENT_TS'].replace('Z', '+00:00'))
					if current_time is not None:
						elapsed_hours = (current_time - in_time).total_seconds() / 3600.0
						progress = elapsed_hours / predicted_cdt_hours if predicted_cdt_hours > 0 else 0
						sample_logs.append(f"  {key}: progress={progress:.1%} color=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})")
				except:
					pass
			
			# Update the vertex colors in the geometry
			try:
				geom_node = node_path.node()
				if isinstance(geom_node, GeomNode) and geom_node.getNumGeoms() > 0:
					for i in range(geom_node.getNumGeoms()):
						geom = geom_node.modifyGeom(i)
						vdata = geom.modifyVertexData()
						
						# Update all vertex colors
						vcolor = GeomVertexWriter(vdata, "color")
						num_vertices = vdata.getNumRows()
						for v in range(num_vertices):
							vcolor.setRow(v)
							vcolor.setData4f(color[0], color[1], color[2], color[3])
						colors_updated += 1
			except Exception as e:
				# If color update fails, continue with other containers
				print(f"Failed to update color for {key}: {e}")
				pass

		if colors_updated > 0:
			print(f"Updated colors for {colors_updated} containers at sim time: {self.current_sim_time}")
			for log in sample_logs:
				print(log)
	
	# Animation methods
	def animate_container_movement(self, container_key: str, from_pos: Tuple, to_pos: Tuple, duration: float = 1):
		"""Animate a container moving from one position to another with detailed effects."""
		if container_key not in self.container_nodes:
			return
		
		node = self.container_nodes[container_key]
		from_x, from_y, from_z = from_pos
		to_x, to_y, to_z = to_pos
		
		# Create position interval (lerp animation)
		from panda3d.core import LVecBase3f, LVecBase4f
		from direct.interval.IntervalGlobal import LerpPosInterval, LerpHprInterval, LerpScaleInterval, LerpColorScaleInterval, Sequence, Parallel
		
		# Lift up, move, then place down with rotation and scale effects
		lift_height = 3.0  # Higher lift for more dramatic effect
		current_pos = node.getPos()
		current_hpr = node.getHpr()
		original_scale = node.getColorScale()

		# Highlight effect
		highlight_color = LVecBase4f(2.0, 2.0, 0.0, 1.0) # Bright yellow
		highlight_on = LerpColorScaleInterval(node, duration * 0.1, highlight_color, blendType='easeOut')
		highlight_off = LerpColorScaleInterval(node, duration * 0.1, original_scale, blendType='easeIn')
		
		# Lift up phase with slight rotation and scale increase
		lift_up_pos = LerpPosInterval(node, duration * 0.25, 
									LVecBase3f(current_pos.x, current_pos.y, current_pos.z + lift_height),
									blendType='easeInOut')
		lift_up_rotate = LerpHprInterval(node, duration * 0.25,
										(current_hpr.x + 5, current_hpr.y, current_hpr.z),
										blendType='easeInOut')
		lift_up_scale = LerpScaleInterval(node, duration * 0.25,
										 (1.05, 1.05, 1.05),
										 blendType='easeInOut')
		lift_up = Parallel(lift_up_pos, lift_up_rotate, lift_up_scale)
		
		# Move across phase
		move_across_pos = LerpPosInterval(node, duration * 0.5,
									   LVecBase3f(to_x, to_y, to_z + lift_height),
									   blendType='easeInOut')
		move_across_rotate = LerpHprInterval(node, duration * 0.5,
											(current_hpr.x, current_hpr.y, current_hpr.z),
											blendType='easeInOut')
		move_across = Parallel(move_across_pos, move_across_rotate)
		
		# Place down phase with scale return
		place_down_pos = LerpPosInterval(node, duration * 0.25,
									  LVecBase3f(to_x, to_y, to_z),
									  blendType='easeInOut')
		place_down_scale = LerpScaleInterval(node, duration * 0.25,
											(1.0, 1.0, 1.0),
											blendType='easeInOut')
		place_down = Parallel(place_down_pos, place_down_scale)
		
		sequence = Sequence(highlight_on, lift_up, move_across, place_down, highlight_off)
		return sequence
	
	def animate_container_removal(self, container_key: str, duration: float = 1.5):
		"""Animate a container being removed - flying away effect."""
		if container_key not in self.container_nodes:
			print(f"WARNING: Container {container_key} not found in container_nodes for removal animation")
			print(f"Available containers: {list(self.container_nodes.keys())[:5]}...")
			return None
		
		node = self.container_nodes[container_key]
		print(f"Starting removal animation for {container_key} at position {node.getPos()}")
		
		from direct.interval.IntervalGlobal import LerpColorScaleInterval, LerpPosInterval, LerpHprInterval, LerpScaleInterval, Parallel, Sequence
		from panda3d.core import LVecBase3f, LVecBase4f
		
		current_pos = node.getPos()
		current_hpr = node.getHpr()
		
		# Highlight effect (Red flash for removal)
		highlight_color = LVecBase4f(2.0, 0.5, 0.5, 1.0) # Bright red
		highlight_on = LerpColorScaleInterval(node, duration * 0.1, highlight_color, blendType='easeOut')

		# If EDI mode is on, turn black immediately
		start_color_anim = Sequence()
		if self.edi_color_mode:
			start_color_anim = LerpColorScaleInterval(node, 0.1, LVecBase4f(0.1, 0.1, 0.1, 1.0), blendType='easeOut')
		else:
			start_color_anim = highlight_on
		
		# Phase 1: Lift up high
		lift_height = 20.0
		lift_duration = duration * 0.3
		lift_up = LerpPosInterval(node, lift_duration, 
								   LVecBase3f(current_pos.x, current_pos.y, current_pos.z + lift_height),
								   blendType='easeOut')
		
		phase1 = Parallel(lift_up, start_color_anim)
		
		# Phase 2: Fly away rapidly to the sky
		fly_duration = duration * 0.7
		
		# Determine fly direction (away from center)
		# Just fly up and towards the "sea" (negative Y) or just away
		target_x = current_pos.x
		target_y = current_pos.y - 200 # Fly towards the "sea" / port side
		target_z = current_pos.z + 100 # Fly high up
		
		fly_away = LerpPosInterval(node, fly_duration,
								   LVecBase3f(target_x, target_y, target_z),
								   blendType='easeIn')
		
		# Rotate while flying for effect
		fly_rotate = LerpHprInterval(node, fly_duration,
									 (current_hpr.x + 180, current_hpr.y + 30, current_hpr.z),
									 blendType='easeInOut')
		
		# Scale down slightly as it flies away to simulate distance
		fly_scale = LerpScaleInterval(node, fly_duration,
									  (0.1, 0.1, 0.1),
									  blendType='easeIn')

		phase2 = Parallel(fly_away, fly_rotate, fly_scale)
		
		removal_anim = Sequence(phase1, phase2)
		return removal_anim
	
	def process_animation_queue(self):
		"""Process queued animations sequentially."""
		if not self.animation_queue or self.is_animating:
			return
		
		self.is_animating = True
		current_animation = self.animation_queue.pop(0)
		
		# Set up callback to process next animation when current one finishes
		def on_animation_complete():
			self.is_animating = False
			if self.animation_queue:
				self.process_animation_queue()
		
		if current_animation:
			current_animation.setDoneEvent('animation_complete')
			self.accept('animation_complete', on_animation_complete)
			current_animation.start()
	
	# Event tracking methods
	def mark_container_event(self, container_key: str, event_type: str, duration_frames: int = 1):
		"""Mark a container with an event (in, update, remove, rehandling).
		
		Args:
			container_key: The unique key of the container
			event_type: Type of event ('in', 'update', 'remove', 'rehandling')
			duration_frames: How long the color should be displayed (in event frames, not time frames)
		"""
		self.container_events[container_key] = {
			'event_type': event_type,
			'frame_count': self.event_frame_counter,
			'duration_frames': duration_frames
		}
	
	def _event_update_task(self, task):
		"""Task to check if any container events have expired and rebuild if needed."""
		# Check if any events have expired (based on frame count)
		expired_keys = []
		for key, event_info in self.container_events.items():
			frames_elapsed = self.event_frame_counter - event_info['frame_count']
			if frames_elapsed >= event_info['duration_frames']:
				expired_keys.append(key)
		
		# If any events expired, rebuild the scene to update colors
		if expired_keys:
			for key in expired_keys:
				del self.container_events[key]
			self.refresh_scene()
		
		return Task.cont
	
	def animate_removal(self, container_key: str, from_pos: Tuple[int, int, int, int, str]):
		"""Animate container removal by moving it up and fading out."""
		if container_key in self.container_nodes:
			node = self.container_nodes[container_key]
			current_pos = node.getPos()
			
			# Animate upward and fade out over 0.5 seconds
			from direct.interval.IntervalGlobal import Sequence, Parallel, LerpPosInterval, LerpScaleInterval, LerpColorScaleInterval
			
			lift_anim = LerpPosInterval(
				node, 
				duration=0.5, 
				pos=(current_pos[0], current_pos[1], current_pos[2] + 5),
				blendType='easeIn'
			)
			
			fade_anim = LerpColorScaleInterval(
				node,
				duration=0.5,
				colorScale=(1, 1, 1, 0),
				blendType='easeIn'
			)
			
			remove_seq = Parallel(lift_anim, fade_anim)
			remove_seq.start()
			
			return remove_seq
		return None
	
	def animate_rehandling(self, container_key: str, from_pos: Tuple[int, int, int, int, str], 
	                       to_pos: Tuple[int, int, int, int, str]):
		"""Animate container rehandling by moving it from one position to another."""
		if container_key in self.container_nodes:
			node = self.container_nodes[container_key]
			
			block_from, x_from, y_from, z_from, yard_type_from = from_pos
			block_to, x_to, y_to, z_to, yard_type_to = to_pos
			
			w, h = self.cfg.width, self.cfg.height
			offset_x_from = block_from * (w + 5)
			offset_y_from = (h + 5) if yard_type_from == 'RF' else 0
			
			offset_x_to = block_to * (w + 5)
			offset_y_to = (h + 5) if yard_type_to == 'RF' else 0
			
			start_pos = (offset_x_from + x_from, offset_y_from + y_from, z_from)
			end_pos = (offset_x_to + x_to, offset_y_to + y_to, z_to)
			lift_height = max(z_from, z_to) + 3
			
			# Create arc movement: up -> move -> down
			from direct.interval.IntervalGlobal import Sequence, LerpPosInterval
			
			lift_up = LerpPosInterval(
				node,
				duration=0.3,
				pos=(start_pos[0], start_pos[1], lift_height),
				blendType='easeOut'
			)
			
			move_across = LerpPosInterval(
				node,
				duration=0.5,
				pos=(end_pos[0], end_pos[1], lift_height),
				blendType='noBlend'
			)
			
			lift_down = LerpPosInterval(
				node,
				duration=0.3,
				pos=end_pos,
				blendType='easeIn'
			)
			
			rehandling_seq = Sequence(lift_up, move_across, lift_down)
			rehandling_seq.start()
			
			return rehandling_seq
		return None
	
	# Mouse and keyboard controls
	def _on_mouse_down(self):
		self.mouse_down = True
		if self.mouseWatcherNode.hasMouse():
			self.last_mouse_x = self.mouseWatcherNode.getMouseX()
			self.last_mouse_y = self.mouseWatcherNode.getMouseY()
			self.click_start_pos = (self.last_mouse_x, self.last_mouse_y)
	
	def _on_mouse_up(self):
		self.mouse_down = False
		
		# Check if this was a click (not a drag)
		if self.mouseWatcherNode.hasMouse() and self.click_start_pos:
			mx = self.mouseWatcherNode.getMouseX()
			my = self.mouseWatcherNode.getMouseY()
			start_x, start_y = self.click_start_pos
			
			# If mouse didn't move much, treat as click
			distance = ((mx - start_x) ** 2 + (my - start_y) ** 2) ** 0.5
			if distance < 0.01:  # Small threshold for click detection
				self._on_container_click(mx, my)
		
		self.click_start_pos = None
	
	def _on_container_click(self, mouse_x, mouse_y):
		"""Handle click on a container to show its information."""
		from panda3d.core import CollisionTraverser, CollisionNode, CollisionRay, CollisionHandlerQueue
		
		print(f"Container click detected at ({mouse_x}, {mouse_y})")  # Debug
		
		# Create a ray from the camera through the mouse position
		picker_ray = CollisionRay()
		picker_ray.setFromLens(self.camNode, mouse_x, mouse_y)
		
		# Create collision traverser
		picker_node = CollisionNode('mouse_ray')
		picker_node.addSolid(picker_ray)
		picker_node.setFromCollideMask(1)  # Match container collision mask
		picker_np = self.camera.attachNewNode(picker_node)
		
		queue = CollisionHandlerQueue()
		traverser = CollisionTraverser('traverser')
		traverser.addCollider(picker_np, queue)
		traverser.traverse(self.render)
		
		print(f"Collision entries: {queue.getNumEntries()}")  # Debug
		
		# Check if we hit something
		if queue.getNumEntries() > 0:
			queue.sortEntries()
			entry = queue.getEntry(0)
			hit_node = entry.getIntoNodePath()
			
			print(f"Hit node: {hit_node.getName()}")  # Debug
			
			# Find which container was clicked by checking parent nodes
			current = hit_node
			for _ in range(5):  # Check up to 5 levels up in hierarchy
				for container_key, node_path in self.container_nodes.items():
					if not container_key.startswith('_animating_'):
						if current == node_path or current.getParent() == node_path:
							print(f"Found container: {container_key}")  # Debug
							self._show_container_info(container_key)
							picker_np.removeNode()
							return
				if current.hasParent():
					current = current.getParent()
				else:
					break
		
		print("No container hit")  # Debug
		# If no container hit, hide the info panel
		self.container_info_np.hide()
		self.selected_container_key = None
		
		picker_np.removeNode()
	
	def _show_container_info(self, container_key: str):
		"""Display information about the selected container."""
		self.selected_container_key = container_key
		
		# Get container information
		container_info = self.yard.containers.get(container_key)
		if not container_info:
			self.container_info_np.hide()
			return
		
		# Get container position
		position_info = "Position: Unknown"
		if container_key in self.yard.container_positions:
			block, x, y, z, yard_type = self.yard.container_positions[container_key]
			position_info = f"Position: Block {block}, ({x},{y},{z}) {yard_type}"
		
		# Format container information
		from datetime import datetime
		
		info_lines = [
			f"=== CONTAINER INFO ===",
			f"Key: {container_key}",
			f"Consignee: {container_info.get('consignee_name', 'N/A')}",
			f"Size: {container_info.get('SZTP2', 'N/A')}",
            f"STATUS: {container_info.get('event_history', 'N/A')[-1]}",
			position_info,
		]
		
		# Add predicted and actual CDT if available
		# Try both std_CDT_pred and no_std_CDT_pred
		pred_field = None
		if 'std_CDT_pred' in container_info:
			pred_field = 'std_CDT_pred'
		elif 'no_std_CDT_pred' in container_info:
			pred_field = 'no_std_CDT_pred'
		
		if pred_field:
			try:
				pred_cdt = container_info[pred_field]
				if isinstance(pred_cdt, (int, float)):
					# CDT is in hours
					pred_str = f"{pred_cdt:.2f} hours"
				elif isinstance(pred_cdt, str):
					dt = datetime.fromisoformat(pred_cdt.replace('Z', '+00:00'))
					pred_str = dt.strftime("%Y-%m-%d %H:%M")
				else:
					pred_str = str(pred_cdt)
				info_lines.append(f"Predicted CDT: {pred_str}")
				
				# Calculate and show CDT progress
				if isinstance(pred_cdt, (int, float)) and 'EVENT_TS' in container_info and self.current_sim_time:
					try:
						from datetime import timedelta
						pred_hours = float(pred_cdt)
						in_time_str = container_info['EVENT_TS']
						in_time = datetime.fromisoformat(in_time_str.replace('Z', '+00:00'))
						
						if isinstance(self.current_sim_time, str):
							current_time = datetime.fromisoformat(self.current_sim_time.replace('Z', '+00:00'))
						else:
							current_time = self.current_sim_time
						
						elapsed_hours = (current_time - in_time).total_seconds() / 3600.0
						progress = (elapsed_hours / pred_hours) * 100 if pred_hours > 0 else 0
						
						info_lines.append(f"CDT Progress: {progress:.1f}% ({elapsed_hours:.1f}h / {pred_hours:.1f}h)")
					except:
						pass
			except Exception as e:
				info_lines.append(f"Predicted CDT: {container_info.get(pred_field, 'N/A')}")
		
		# Add Actual CDT if available (case-insensitive check)
		cdt_true_key = next((k for k in container_info.keys() if k.lower() == 'cdt_true'), None)
		print("CDT true key found:", cdt_true_key)  # Debug	
		if cdt_true_key:
			try:
				from datetime import datetime
				true_cdt = container_info[cdt_true_key]
				
				if pd.isna(true_cdt):
					true_str = "N/A"
				elif isinstance(true_cdt, (int, float)):
					# CDT is in hours
					true_str = f"{true_cdt:.2f} hours"
				elif isinstance(true_cdt, str):
					dt = datetime.fromisoformat(true_cdt.replace('Z', '+00:00'))
					true_str = dt.strftime("%Y-%m-%d %H:%M")
				else:
					true_str = str(true_cdt)
				info_lines.append(f"Actual CDT: {true_str}")
			except Exception as e:
				val = container_info.get(cdt_true_key, 'N/A')
				info_lines.append(f"Actual CDT: {val}")
		
		# Add event timestamp
		if 'EVENT_TS' in container_info:
			try:
				event_ts = container_info['EVENT_TS']
				if isinstance(event_ts, str):
					dt = datetime.fromisoformat(event_ts.replace('Z', '+00:00'))
					ts_str = dt.strftime("%Y-%m-%d %H:%M")
				else:
					ts_str = str(event_ts)
				info_lines.append(f"IN Time: {ts_str}")
			except:
				info_lines.append(f"IN Time: {container_info.get('EVENT_TS', 'N/A')}")
		

		
		info_lines.append("(Click elsewhere to hide)")
		
		# Set text and show
		info_text = "\n".join(info_lines)
		self.container_info_text.setText(info_text)
		self.container_info_np.show()
		
		print(f"Showing info for container: {container_key}")
	
	def _mouse_task(self, task):
		if self.mouse_down and self.mouseWatcherNode.hasMouse():
			mx = self.mouseWatcherNode.getMouseX()
			my = self.mouseWatcherNode.getMouseY()
			dx = mx - self.last_mouse_x
			dy = my - self.last_mouse_y
			
			# Rotate camera
			self.camera_yaw += dx * 100
			self.camera_pitch += dy * 100
			self.camera_pitch = max(-89, min(89, self.camera_pitch))
			
			self._update_camera_position()
			
			self.last_mouse_x = mx
			self.last_mouse_y = my
		
		return Task.cont
	
	def _on_zoom_in(self):
		self.camera_distance *= 0.9
		self.camera_distance = max(5, self.camera_distance)
		self._update_camera_position()
	
	def _on_zoom_out(self):
		self.camera_distance *= 1.1
		self.camera_distance = min(500, self.camera_distance)
		self._update_camera_position()
	
	def _rotate_left(self):
		self.camera_yaw -= 5
		self._update_camera_position()
	
	def _rotate_right(self):
		self.camera_yaw += 5
		self._update_camera_position()
	
	def _rotate_up(self):
		self.camera_pitch += 5
		self.camera_pitch = min(89, self.camera_pitch)
		self._update_camera_position()
	
	def _rotate_down(self):
		self.camera_pitch -= 5
		self.camera_pitch = max(-89, self.camera_pitch)
		self._update_camera_position()
	
	def _set_preset_view(self, view_name: str):
		"""Set camera to a preset viewing angle."""
		presets = {
			"NE": (45, -35),
			"SE": (135, -35),
			"SW": (225, -35),
			"NW": (315, -35),
			"TOP": (45, -89),     # Top-down view
			"FRONT": (0, 0),      # Front view
		}
		
		if view_name in presets:
			self.camera_yaw, self.camera_pitch = presets[view_name]
			self._update_camera_position()
			print(f"View changed to: {view_name}")
	
	def _reset_view(self):
		"""Reset camera to default position and zoom."""
		self.camera_yaw = 45
		self.camera_pitch = -45
		
		# Calculate layout dimensions for rows of 10 blocks
		BLOCKS_PER_ROW = 10
		total_rows = (self.total_blocks + BLOCKS_PER_ROW - 1) // BLOCKS_PER_ROW
		layout_width = min(self.total_blocks, BLOCKS_PER_ROW) * (self.cfg.width + 5)
		layout_height = total_rows * ((self.cfg.height + 5) * 2 + 10)
		
		self.camera_distance = max(layout_width, layout_height) * 1.2
		
		# Reset focus to layout center
		center_x = layout_width / 2
		center_y = layout_height / 2
		center_z = self.cfg.depth / 2
		self.camera_focus = [center_x, center_y, center_z]
		self._update_camera_position()
		print("View reset to default")
	
	def _on_ctrl_click(self):
		"""Handle Ctrl + Click to change camera focus point."""
		if not self.mouseWatcherNode.hasMouse():
			return
		
		# Get mouse position in normalized coordinates (-1 to 1)
		mouse_x = self.mouseWatcherNode.getMouseX()
		mouse_y = self.mouseWatcherNode.getMouseY()
		
		# Perform ray casting from camera through mouse position
		from panda3d.core import CollisionTraverser, CollisionNode, CollisionRay, CollisionHandlerQueue
		from panda3d.core import Point3, Vec3
		
		# Create a ray from the camera through the mouse position
		picker_ray = CollisionRay()
		picker_ray.setFromLens(self.camNode, mouse_x, mouse_y)
		
		# Create collision traverser
		picker_node = CollisionNode('mouse_ray')
		picker_node.addSolid(picker_ray)
		picker_node.setFromCollideMask(0)
		picker_np = self.camera.attachNewNode(picker_node)
		
		queue = CollisionHandlerQueue()
		traverser = CollisionTraverser('traverser')
		traverser.addCollider(picker_np, queue)
		traverser.traverse(self.render)
		
		# Check if we hit something
		if queue.getNumEntries() > 0:
			queue.sortEntries()
			entry = queue.getEntry(0)
			hit_pos = entry.getSurfacePoint(self.render)
			
			# Set new focus point
			self.camera_focus = [hit_pos.getX(), hit_pos.getY(), hit_pos.getZ()]
			self._update_camera_position()
			print(f"Camera focus changed to: ({hit_pos.getX():.1f}, {hit_pos.getY():.1f}, {hit_pos.getZ():.1f})")
		else:
			# If no hit, try to project onto ground plane (z=0)
			# Get camera position and direction
			cam_pos = self.camera.getPos()
			
			# Calculate ray direction from camera through mouse
			near_point = Point3()
			far_point = Point3()
			self.camLens.extrude((mouse_x, mouse_y), near_point, far_point)
			
			# Transform to world space
			near_world = self.render.getRelativePoint(self.camera, near_point)
			far_world = self.render.getRelativePoint(self.camera, far_point)
			
			# Calculate intersection with z=0 plane
			ray_dir = far_world - near_world
			ray_dir.normalize()
			
			# If ray is pointing somewhat downward, intersect with ground
			if ray_dir.getZ() < 0:
				t = -near_world.getZ() / ray_dir.getZ()
				hit_x = near_world.getX() + t * ray_dir.getX()
				hit_y = near_world.getY() + t * ray_dir.getY()
				
				# Set new focus point on ground
				self.camera_focus = [hit_x, hit_y, 0]
				self._update_camera_position()
				print(f"Camera focus changed to ground: ({hit_x:.1f}, {hit_y:.1f}, 0.0)")
		
		# Clean up
		picker_np.removeNode()
	
	def user_exit(self):
		"""Exit the application."""
		self.userExit()
	
	def _toggle_edi_color_mode(self):
		"""Toggle EDI color mode."""
		self.edi_color_mode = not self.edi_color_mode
		mode_status = "ON" if self.edi_color_mode else "OFF"
		print(f"EDI Color Mode: {mode_status}")
		
		# Update HUD to show mode status
		if self.edi_color_mode:
			self.hud_text.setText(f"{self.hud_text.getText()}\nEDI Mode: ON")
		else:
			# Remove EDI Mode text if present
			current_text = self.hud_text.getText()
			if "\nEDI Mode: ON" in current_text:
				self.hud_text.setText(current_text.replace("\nEDI Mode: ON", ""))
		
		# Refresh all container colors
		self.update_container_colors()
	
	def _toggle_decorations_and_text(self):
		"""Toggle visibility of decorations (ocean, fence, lighting poles, office, gate), yard labels (GP0, RF0), and text HUDs."""
		self.show_decorations_and_text = not self.show_decorations_and_text
		self._apply_decorations_visibility()
		
		if self.show_decorations_and_text:
			print("Decorations and text: ON")
		else:
			print("Decorations and text: OFF")
	
	def _apply_decorations_visibility(self):
		"""Apply the current visibility state to decorations, labels, and HUD elements."""
		# Check if HUD elements exist (they may not exist during initial scene build)
		hud_exists = hasattr(self, 'hud_np')
		camera_pos_exists = hasattr(self, 'camera_pos_np')
		
		if self.show_decorations_and_text:
			# Show decorations and text
			self.port_root.show()  # Ocean
			self.decorations_root.show()  # Fence, lighting, office, gate
			self.yard_labels_root.show()  # GP0, RF0, etc.
			if hud_exists:
				self.hud_np.show()
				self.next_container_np.show()
				self.sim_time_np.show()
				# container_info_np is handled separately (only shown when container is selected)
				if self.selected_container_key:
					self.container_info_np.show()
			if camera_pos_exists:
				self.camera_pos_np.show()  # Camera position
		else:
			# Hide decorations and text
			self.port_root.hide()  # Ocean
			self.decorations_root.hide()  # Fence, lighting, office, gate
			self.yard_labels_root.hide()  # GP0, RF0, etc.
			if hud_exists:
				self.hud_np.hide()
				self.next_container_np.hide()
				self.sim_time_np.hide()
				self.container_info_np.hide()
			if camera_pos_exists:
				self.camera_pos_np.hide()  # Camera position


# Keep the old PygameYardRenderer3D class name as an alias for compatibility
PygameYardRenderer3D = Panda3DYardRenderer


# ====================== Realtime Panda3D Play Function ====================== #
def play_in_panda3d(
	df: pd.DataFrame,
	yard_config: YardSimConfig,
	yard: Optional[YardSimulation] = None,
	strategy_cls=None,
	ms_per_event: int = 20,
	blocks_per_row: int = 3,
	lane_every: Optional[int] = None,
	colorful: bool = True,
	window_title: str = "Yard Simulation (Panda3D)",
	window_size: Tuple[int, int] = (1280, 720),
	hud_title: str = "",
	view_orientation: str = "NE",
):
	"""Open a Panda3D window and animate the simulation processing events.

	Controls:
	- ESC: quit
	- Arrow keys or WASD: rotate camera
	- Mouse wheel: zoom in/out
	- Left Mouse Drag: rotate view
	- SPACE: pause/resume simulation
	
	Args:
		df: DataFrame with simulation events
		yard_config: Yard configuration
		yard: Optional pre-initialized yard simulation
		strategy_cls: Simulation class to use
		ms_per_event: Milliseconds between processing events
		blocks_per_row: Number of blocks per row in layout
		lane_every: Draw lane markers every N rows
		colorful: Use colorful containers
		window_title: Window title
		window_size: Window size (width, height)
		hud_title: Title to display in HUD
		view_orientation: Initial camera orientation (NE/SE/SW/NW)
	"""
	if ShowBase is None:
		raise ImportError("Panda3D is required. Install with `pip install panda3d`.")

	# Prepare simulation
	sim = yard or YardSimulation(yard_config=yard_config or YardSimConfig(), stacking_strategy=strategy_cls(cdt_key=(yard_config.cdt_key if yard_config else 'CDT_true')) if strategy_cls else None)
	lane_n = (sim.gp_blocks // 3) if lane_every is None else int(lane_every)

	# Sort events
	if isinstance(df, pd.DataFrame):
		events = df.sort_values("EVENT_TS").to_dict(orient="records")
	else:
		events = list(df)

	# Build renderer
	renderer = Panda3DYardRenderer(
		sim,
		config=yard_config,
		blocks_per_row=blocks_per_row,
		lane_every=lane_n,
		colorful=colorful,
		hud_title=hud_title,
		window_title=window_title,
		window_size=window_size,
	)
	
	# Set initial camera orientation based on view_orientation
	orientation_map = {
		"NE": (45, -35),
		"SE": (135, -35),
		"SW": (225, -35),
		"NW": (315, -35),
	}
	yaw, pitch = orientation_map.get(view_orientation.upper(), (45, -35))
	renderer.camera_yaw = yaw
	renderer.camera_pitch = pitch
	renderer._update_camera_position()

	# Event processing state
	event_idx = [0]  # Use list to allow modification in task
	paused = [False]
	delay_seconds = [ms_per_event / 1000.0]  # Use list to allow modification
	speed_multiplier = [1.0]  # Speed multiplier for simulation
	last_event_time = [0.0]
	
	def update_next_container_info():
		"""Update the HUD with next container information."""
		if event_idx[0] < len(events):
			next_event = events[event_idx[0]]
			event_type = next_event.get('EVENT_TYPE', '')
			
			# Set color based on event type
			if event_type == 'IN':
				# Green for IN events
				renderer.next_container_text.setTextColor(0.2, 0.8, 0.2, 1)
				# Show info for next incoming container
				unique_key = next_event.get('UNIQUE_KEY', 'N/A')
				consignee = next_event.get('consignee_name', 'N/A')
				size = next_event.get('CARGO_SIZE', next_event.get('SZTP2', 'N/A'))
				event_ts = next_event.get('EVENT_TS', 'N/A')
				
				# Truncate long consignee names
				if len(consignee) > 30:
					consignee = consignee[:27] + '...'
				
				info_text = f"NEXT IN:\n{unique_key}\n{consignee}\nSize: {size}\nTime: {event_ts}"
				renderer.next_container_text.setText(info_text)
			elif event_type == 'OUT':
				# Red for OUT events
				renderer.next_container_text.setTextColor(0.9, 0.2, 0.2, 1)
				unique_key = next_event.get('UNIQUE_KEY', 'N/A')
				info_text = f"NEXT {event_type}:\n{unique_key}"
				renderer.next_container_text.setText(info_text)
				# Mark the container that will be removed in red
				renderer.mark_container_event(unique_key, 'remove', duration_frames=999)
			else:
				# Blue for UPDATE events (CUS, COR, COP)
				renderer.next_container_text.setTextColor(0.2, 0.5, 1.0, 1)
				unique_key = next_event.get('UNIQUE_KEY', 'N/A')
				info_text = f"NEXT {event_type}:\n{unique_key}"
				renderer.next_container_text.setText(info_text)
		else:
			renderer.next_container_text.setTextColor(1, 1, 1, 1)  # White for completion
			renderer.next_container_text.setText("SIMULATION COMPLETE")

	def process_event_task(task):
		"""Task that processes simulation events over time."""
        # Update next container info
		update_next_container_info()
  
		if paused[0]:
			return Task.cont
		
		# Wait if animation is in progress
		if renderer.is_animating:
			return Task.cont
		
		current_time = task.time
		# Calculate actual delay with speed multiplier
		actual_delay = delay_seconds[0] / speed_multiplier[0]
		if event_idx[0] < len(events) and (current_time - last_event_time[0]) >= actual_delay:
			event = events[event_idx[0]]
			event_type = event.get('EVENT_TYPE', '')
			unique_key = event.get('UNIQUE_KEY', '')
			
			# Update current simulation time for dwell time calculation
			if 'EVENT_TS' in event:
				renderer.current_sim_time = current_time
				# Update simulation time display
				from datetime import datetime
				try:
					if isinstance(current_time, str):
						time_str = current_time
						dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
						formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
					else:
						formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
					renderer.sim_time_text.setText(formatted_time)
				except:
					renderer.sim_time_text.setText(str(event['EVENT_TS']))
			
			# Increment event frame counter
			renderer.event_frame_counter += 1
			
			# Save old positions before processing event (for animations)
			old_positions = {}
			rehandled_containers = []
			
			# Prepare animations BEFORE processing the event
			from direct.interval.IntervalGlobal import Sequence, Parallel
			rehandling_animations = []
			removal_animation = None
			
			if event_type == 'OUT':
				# Track positions of containers that might be rehandled
				if unique_key in sim.container_positions:
					target_pos = sim.container_positions[unique_key]
					block, x, y, z, yard_type = target_pos
					yard = sim.rf_yard if yard_type == 'RF' else sim.gp_yard
					w, h, d = sim.yard_width, sim.yard_height, sim.yard_depth
					offset_x = block * (w + 5)
					offset_y = (h + 5) if yard_type == 'RF' else 0
					
					# Find containers above the target
					for above_z in range(z + 1, sim.yard_depth):
						above_key = yard[block, x, y, above_z]
						if above_key and '_2nd' not in str(above_key):
							if above_key in sim.container_positions:
								old_pos = sim.container_positions[above_key]
								old_b, old_x, old_y, old_z, old_type = old_pos
								old_offset_x = old_b * (w + 5)
								old_offset_y = (h + 5) if old_type == 'RF' else 0
								old_positions[above_key] = (old_offset_x + old_x, old_offset_y + old_y, old_z)
								rehandled_containers.append(above_key)
					
					# Also save the OUT container position
					old_positions[unique_key] = (offset_x + x, offset_y + y, z)
					
					# IMPORTANT: Save the node reference BEFORE processing event
					# This preserves the node for animation even after refresh_scene() clears container_nodes
					if unique_key in renderer.container_nodes:
						out_node = renderer.container_nodes[unique_key]
						# Store it temporarily with a special key
						renderer.container_nodes[f"_animating_{unique_key}"] = out_node
						print(f"Saved OUT container node {unique_key} for animation")
					
					# Create OUT removal animation BEFORE processing event
					renderer.mark_container_event(unique_key, 'remove', duration_frames=10)
			
			# Process the simulation event
			sim.process_event(event)
			
			# Create removal animation AFTER processing (using saved node)
			if event_type == 'OUT' and unique_key in old_positions:
				temp_key = f"_animating_{unique_key}"
				print(f"Checking for temp key: {temp_key}")
				print(f"Available keys: {[k for k in renderer.container_nodes.keys() if '_animating_' in k]}")
				if temp_key in renderer.container_nodes:
					# Temporarily restore the key for animation
					renderer.container_nodes[unique_key] = renderer.container_nodes[temp_key]
					print(f"Creating removal animation for {unique_key}...")
					# removal_animation = renderer.animate_container_removal(unique_key, duration=2.0)
					removal_animation = None # Disable removal animation
					# print(f"Created removal animation for {unique_key}: {removal_animation is not None}")
					# DON'T delete temp key yet - keep it until animation completes
					if not removal_animation:
						# Animation failed, clean up
						# del renderer.container_nodes[temp_key]
						# print(f"WARNING: Failed to create removal animation for {unique_key}")
						pass
				else:
					# print(f"WARNING: Temp key {temp_key} not found in container_nodes")
					pass
			
			# Handle animations for REHANDLING after processing
			if event_type == 'OUT':
				# Animate rehandled containers if any
				if hasattr(sim, 'last_rehandled_containers') and sim.last_rehandled_containers:
					for rehandled_key in sim.last_rehandled_containers:
						if rehandled_key in old_positions and rehandled_key in sim.container_positions:
							# Mark as rehandling
							renderer.mark_container_event(rehandled_key, 'rehandling', duration_frames=10)
							
							# Get new position
							new_pos = sim.container_positions[rehandled_key]
							new_b, new_x, new_y, new_z, new_type = new_pos
							w, h, d = sim.yard_width, sim.yard_height, sim.yard_depth
							new_offset_x = new_b * (w + 5)
							new_offset_y = (h + 5) if new_type == 'RF' else 0
							new_world_pos = (new_offset_x + new_x, new_offset_y + new_y, new_z)
							
							# Create animation
							anim = renderer.animate_container_movement(
								rehandled_key, 
								old_positions[rehandled_key], 
								new_world_pos,
								duration=1.0
							)
							if anim:
								rehandling_animations.append(anim)
			
			# Execute animations
			if rehandling_animations or removal_animation:
				print(f"Executing animations - removal: {removal_animation is not None}, rehandling: {len(rehandling_animations)}")
				
				from direct.interval.IntervalGlobal import Sequence, Parallel, Func
				
				renderer.is_animating = True
				full_sequence = Sequence()
				
				# 1. Rehandling Animations (First, clear the way)
				if rehandling_animations:
					rehandling_parallel = Parallel(*rehandling_animations)
					full_sequence.append(rehandling_parallel)
				
				# 2. Removal Animation (Then remove the target)
				if removal_animation:
					temp_key = f"_animating_{unique_key}"
					
					def cleanup_removal():
						if temp_key in renderer.container_nodes:
							del renderer.container_nodes[temp_key]
						if unique_key in renderer.container_nodes:
							del renderer.container_nodes[unique_key]
						print(f"Removal animation complete for {unique_key}")
					
					full_sequence.append(removal_animation)
					full_sequence.append(Func(cleanup_removal))
				elif event_type == 'OUT':
					# If no animation but it is an OUT event, we still need to remove it after rehandling
					temp_key = f"_animating_{unique_key}"
					def cleanup_removal_instant():
						if temp_key in renderer.container_nodes:
							del renderer.container_nodes[temp_key]
						if unique_key in renderer.container_nodes:
							del renderer.container_nodes[unique_key]
						print(f"Instant removal complete for {unique_key}")
					full_sequence.append(Func(cleanup_removal_instant))
				
				def on_sequence_complete():
					renderer.is_animating = False
					renderer.refresh_scene()
				
				full_sequence.setDoneEvent('anim_sequence_complete')
				renderer.accept('anim_sequence_complete', on_sequence_complete)
				full_sequence.start()
			else:
				# No animations, just mark events
				if event_type == 'IN':
					renderer.mark_container_event(unique_key, 'in', duration_frames=1)
				elif event_type in ['CUS', 'COR', 'COP']:
					renderer.mark_container_event(unique_key, 'update', duration_frames=1)
				
				# Refresh scene immediately if no animation
				renderer.refresh_scene()
			
			event_idx[0] += 1
			last_event_time[0] = task.time
			
			
		
		return Task.cont

	# Add pause/resume control
	def toggle_pause():
		paused[0] = not paused[0]
		print(f"Simulation {'paused' if paused[0] else 'resumed'}")
	
	# Add speed control
	def speed_up():
		speed_multiplier[0] *= 2.0
		speed_multiplier[0] = min(speed_multiplier[0], 32.0)  # Max 32x speed
		print(f"Simulation speed: {speed_multiplier[0]:.1f}x")
	
	def slow_down():
		speed_multiplier[0] /= 2.0
		speed_multiplier[0] = max(speed_multiplier[0], 0.125)  # Min 0.125x (1/8) speed
		print(f"Simulation speed: {speed_multiplier[0]:.1f}x")
	
	renderer.accept("space", toggle_pause)
	renderer.accept("page_up", speed_up)
	renderer.accept("page_down", slow_down)
	
	# Add color update task (refresh scene periodically to update dwell time colors)
	last_color_update = [0.0]
	COLOR_UPDATE_INTERVAL = 1.0  # Update colors every 1 second
	
	def update_colors_task(task):
		"""Refresh container colors periodically based on dwell time."""
		if not paused[0] and not renderer.is_animating:
			if task.time - last_color_update[0] >= COLOR_UPDATE_INTERVAL:
				# Update colors without rebuilding the entire scene
				renderer.update_container_colors()
				last_color_update[0] = task.time
		return Task.cont
	
	# Add event processing task
	renderer.taskMgr.add(process_event_task, "process_events")
	renderer.taskMgr.add(update_colors_task, "update_colors")
	
	# Run the application
	renderer.run()
	
	return sim


# Alias for compatibility
def play_in_ursina(*args, **kwargs):
	"""Alias for play_in_panda3d for backward compatibility."""
	print("Note: play_in_ursina is now using Panda3D implementation")
	return play_in_panda3d(*args, **kwargs)

