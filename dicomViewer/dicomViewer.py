import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.clock import Clock
from kivy.graphics.context_instructions import Color
from kivy.core.window import Window

#for the mask r cnn
import os
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
#from PIL import Image
import torchvision
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split



class Tooltip(Popup):
	def _init_(self, text, **kwargs):
		super(Tooltip, self)._init_(**kwargs)
		self.size_hint = (None, None)
		self.size = (150, 50)
		self.background_color = (0, 0, 0, 0.75)
		self.border = (10, 10, 10, 10)
		self.auto_dismiss = True
		self.label = Label(text=text, color=(1, 1, 1, 1))
		self.add_widget(self.label)

	def open(self, *args, **kwargs):
		super(Tooltip, self).open(*args, **kwargs)
		Clock.schedule_once(self.dismiss, 2)  # Auto dismiss after 2 seconds


'''
class StartScreen(Screen):
	def _init_(self, **kwargs):
		super(StartScreen, self)._init_(**kwargs)
		layout = BoxLayout(orientation='horizontal')

		self.left_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1))
		self.right_panel = BoxLayout(
			orientation='vertical', size_hint=(0.7, 1))

		self.file_chooser = FileChooserIconView(path=os.getcwd())
		self.left_panel.add_widget(self.file_chooser)

		self.file_chooser.bind(
			on_selection=lambda *x: self.show_file_details())

		self.file_details = Label(text="Select a file to see details")
		self.right_panel.add_widget(self.file_details)

		open_button = Button(text='Open DICOM', size_hint=(1, 0.1))
		open_button.bind(on_release=self.open_dicom)
		self.right_panel.add_widget(open_button)

		layout.add_widget(self.left_panel)
		layout.add_widget(self.right_panel)
		self.add_widget(layout)

	def show_file_details(self):
		if self.file_chooser.selection:
			dicom_file = self.file_chooser.selection[0]
			dicom_data = pydicom.dcmread(dicom_file)
			details = f"File: {os.path.basename(dicom_file)}\n"
			details += f"Patient ID: {dicom_data.PatientID}\n"
			details += f"Modality: {dicom_data.Modality}\n"
			details += f"Study Date: {dicom_data.StudyDate}\n"
			self.file_details.text = details

	def open_dicom(self, instance):
		if self.file_chooser.selection:
			dicom_file = self.file_chooser.selection[0]
			self.manager.current = 'viewer'
			self.manager.get_screen('viewer').load_dicom(dicom_file)
'''


class StartScreen(Screen):
	def __init__(self, **kwargs):
		super(StartScreen, self).__init__(**kwargs)
		layout = BoxLayout(orientation='horizontal')

		self.left_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1))
		self.right_panel = BoxLayout(
			orientation='vertical', size_hint=(0.7, 1))

		self.file_chooser = FileChooserIconView(path=os.getcwd())
		self.left_panel.add_widget(self.file_chooser)

		self.file_chooser.bind(
			on_selection=lambda *x: self.show_file_details())

		self.file_details = Label(text="Select a file to see details")
		self.right_panel.add_widget(self.file_details)

		open_button = Button(text='Open DICOM', size_hint=(1, 0.1))
		open_button.bind(on_release=self.open_dicom)
		self.right_panel.add_widget(open_button)

		layout.add_widget(self.left_panel)
		layout.add_widget(self.right_panel)
		self.add_widget(layout)

	def show_file_details(self):
		if self.file_chooser.selection:
			dicom_file = self.file_chooser.selection[0]
			dicom_data = pydicom.dcmread(dicom_file)
			details = f"File: {os.path.basename(dicom_file)}\n"
			details += f"Patient ID: {dicom_data.PatientID}\n"
			details += f"Modality: {dicom_data.Modality}\n"
			details += f"Study Date: {dicom_data.StudyDate}\n"
			self.file_details.text = details

	def open_dicom(self, instance):
		if self.file_chooser.selection:
			dicom_file = self.file_chooser.selection[0]
			self.manager.current = 'viewer'
			self.manager.get_screen('viewer').load_dicom(dicom_file)


'''
class ViewerScreen(Screen):
	def _init_(self, **kwargs):
		super(ViewerScreen, self)._init_(**kwargs)
		self.layout = BoxLayout(orientation='vertical')

		# Toolbar layout
		self.toolbar_layout = BoxLayout(
			size_hint=(1, 0.1), orientation='horizontal')

		# Add icons to the toolbar instead of text buttons
		self.tools = [
			('CT Value', 'icons_black/icon_ct_value.png'),
			('MPR', 'icons_black/icon_mpr.png'),
			('Capture', 'icons_black/icon_capture.png'),
			('Export', 'icons_black/icon_export.png'),
			('Rotate', 'icons_black/icon_rotate.png'),
			('Flip Horizontal', 'icons_black/icon_flip_horizontal.png'),
			('Flip Vertical', 'icons_black/icon_flip_vertical.png'),
			('Annotate', 'icons_black/icon_annotate.png'),
			('AI', 'icons_black/icon_ai.png')
		]

		self.tool_buttons = []
		for tool, icon in self.tools:
			btn = Button(background_normal=icon, size_hint_x=None, width=50)
			btn.bind(on_release=lambda instance,
					 x=tool: self.on_tool_select(x))
			btn.bind(on_enter=lambda instance,
					 x=tool: self.show_tooltip(instance, x))
			btn.bind(on_leave=self.hide_tooltip)
			self.tool_buttons.append(btn)

		# Add New Window button to the toolbar
		new_window_button = Button(
			text='New Window', size_hint_x=None, width=100)
		new_window_button.bind(on_release=self.add_new_window)
		self.toolbar_layout.add_widget(new_window_button)

		# Add all tool buttons
		total_width = new_window_button.width
		visible_tool_buttons = []
		for btn in self.tool_buttons:
			if total_width + btn.width > self.toolbar_layout.width:
				break
			self.toolbar_layout.add_widget(btn)
			total_width += btn.width
			visible_tool_buttons.append(btn)

		# Add remaining tools in dropdown if they don't fit
		remaining_tools = len(self.tool_buttons) - len(visible_tool_buttons)
		if remaining_tools > 0:
			dropdown_button = Button(text='⋮', size_hint_x=None, width=50)
			self.dropdown = DropDown()
			for tool, icon in self.tools[-remaining_tools:]:
				btn = Button(background_normal=icon,
							 size_hint_y=None, height=44)
				btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
				self.dropdown.add_widget(btn)
			dropdown_button.bind(on_release=self.dropdown.open)
			self.toolbar_layout.add_widget(dropdown_button)

		# Add settings button to the right upper corner of the toolbar
		settings_button = Button(text='Settings', size_hint=(0.2, 1))
		settings_button.bind(on_release=self.open_settings)
		self.toolbar_layout.add_widget(settings_button)

		self.layout.add_widget(self.toolbar_layout)

		# Image container with GridLayout for flexible arrangement
		self.image_container = GridLayout(cols=1, spacing=10, size_hint_y=None)
		self.image_container.bind(
			minimum_height=self.image_container.setter('height'))
		self.layout.add_widget(self.image_container)

		self.add_widget(self.layout)
		self.dicom_data = None
		self.image_array = None
		self.num_slices = 0
		self.current_slice = 0
		self.load_initial_window = True  # Flag to load initial window

	def show_tooltip(self, instance, tool):
		self.tooltip = Tooltip(text=tool)
		self.tooltip.open(instance)

	def hide_tooltip(self, instance):
		if self.tooltip:
			self.tooltip.dismiss()

	def on_tool_select(self, tool):
		print(f"Selected tool: {tool}")
		if tool == 'Rotate':
			self.rotate_image()
		elif tool == 'Flip Horizontal':
			self.flip_image(horizontal=True)
		elif tool == 'Flip Vertical':
			self.flip_image(horizontal=False)
		elif tool == 'MPR':
			self.show_mpr()
		elif tool == 'CT Value':
			self.show_ct_value()
		elif tool == 'Capture':
			self.capture_image()
		elif tool == 'Export':
			self.export_image()
		elif tool == 'Annotate':
			self.annotate_image()
		elif tool == 'AI':
			self.predict()

	def load_dicom(self, dicom_file):
		dicom_data = pydicom.dcmread(dicom_file)
		self.image_array = dicom_data.pixel_array
		self.dicom_data = dicom_data

		self.image_container.clear_widgets()
		if len(self.image_array.shape) == 3:  # If the image is a volume
			self.num_slices = self.image_array.shape[0]
			self.current_slice = 0
		else:
			self.num_slices = 1
			self.current_slice = 0

		# Create and add initial image window
		if self.load_initial_window:
			self.add_new_window()

	def display_image(self, image_array, image_viewer, zoom=1):
		if len(image_array.shape) == 2:
			height, width = image_array.shape
		else:
			raise ValueError("Expected a 2D array for display_image")

		zoomed_array = cv2.resize(
			image_array, (int(width * zoom), int(height * zoom)))
		image_texture = self.numpy_to_texture(zoomed_array)
		image_viewer.texture = image_texture

	def numpy_to_texture(self, image_array):
		image_array = np.flipud(image_array)
		buffer = image_array.tobytes()
		texture = Texture.create(
			size=(image_array.shape[1], image_array.shape[0]), colorfmt='luminance')
		texture.blit_buffer(buffer, colorfmt='luminance', bufferfmt='ubyte')
		return texture

	def on_slice_change(self, instance, value):
		self.current_slice = int(value)
		self.display_image(
			self.image_array[self.current_slice], instance.image_viewer)

	def add_new_window(self, instance=None):
		window_layout = BoxLayout(
			orientation='vertical', size_hint_y=None, height=300)
		image_viewer = Image()
		window_layout.add_widget(image_viewer)
		slice_slider = Slider(min=0, max=self.num_slices - 1,
							  value=self.current_slice, step=1)
		slice_slider.image_viewer = image_viewer  # Bind image_viewer to the slider
		slice_slider.bind(value=self.on_slice_change)
		window_layout.add_widget(slice_slider)
		self.image_container.add_widget(window_layout)
		self.display_image(self.image_array[self.current_slice], image_viewer)
		self.image_container.cols = int(
			len(self.image_container.children) ** 0.5)

	def rotate_image(self):
		self.image_array = np.rot90(self.image_array, axes=(1, 2))
		for child in self.image_container.children:
			# child.children[0] is the Image widget
			self.display_image(
				self.image_array[self.current_slice], child.children[0])

	def flip_image(self, horizontal=True):
		if horizontal:
			self.image_array = np.flip(self.image_array, axis=2)
		else:
			self.image_array = np.flip(self.image_array, axis=1)
		for child in self.image_container.children:
			# child.children[0] is the Image widget
			self.display_image(
				self.image_array[self.current_slice], child.children[0])

	def show_mpr(self):
		self.mpr_layout = BoxLayout(orientation='horizontal')

		self.axial_view = Image()
		self.coronal_view = Image()
		self.sagittal_view = Image()

		self.mpr_layout.add_widget(self.axial_view)
		self.mpr_layout.add_widget(self.coronal_view)
		self.mpr_layout.add_widget(self.sagittal_view)

		self.add_widget(self.mpr_layout)

		self.update_mpr_views()

	def update_mpr_views(self):
		axial_image = self.image_array[:, :, self.image_array.shape[2] // 2]
		coronal_image = self.image_array[:, self.image_array.shape[1] // 2, :]
		sagittal_image = self.image_array[self.image_array.shape[0] // 2, :, :]

		self.axial_view.texture = self.numpy_to_texture(axial_image)
		self.coronal_view.texture = self.numpy_to_texture(coronal_image)
		self.sagittal_view.texture = self.numpy_to_texture(sagittal_image)

	def show_ct_value(self):
		for child in self.image_container.children:
			child.children[0].bind(on_touch_down=self.display_ct_value)

	def display_ct_value(self, instance, touch):
		if instance.collide_point(*touch.pos):
			x = int(touch.x * self.image_array.shape[2] / instance.width)
			y = int((instance.height - touch.y) *
					self.image_array.shape[1] / instance.height)
			ct_value = self.image_array[self.current_slice, y, x]
			popup = Popup(title='CT Value',
						  content=Label(text=f'CT Value: {ct_value}'),
						  size_hint=(0.3, 0.3))
			popup.open()

	def capture_image(self):
		plt.imshow(self.image_array[self.current_slice], cmap='gray')
		plt.title('Captured Image')
		plt.show()

	def export_image(self):
		export_popup = Popup(title='Export Image',
							 content=TextInput(hint_text='Enter file name'),
							 size_hint=(0.4, 0.3))
		export_popup.bind(on_dismiss=lambda x: self.save_image(
			export_popup.content.text))
		export_popup.open()

	def save_image(self, filename):
		if filename:
			cv2.imwrite(filename, self.image_array[self.current_slice])

	def annotate_image(self):
		for child in self.image_container.children:
			child.children[0].bind(on_touch_down=self.add_annotation)

	def add_annotation(self, instance, touch):
		if instance.collide_point(*touch.pos):
			x = int(touch.x * self.image_array.shape[2] / instance.width)
			y = int((instance.height - touch.y) *
					self.image_array.shape[1] / instance.height)
			self.image_array[self.current_slice] = cv2.circle(
				self.image_array[self.current_slice], (x, y), radius=10, color=(255, 255, 255), thickness=-1)
			self.display_image(self.image_array[self.current_slice], instance)

	def open_settings(self, instance):
		self.manager.current = 'settings'
'''
class ImageWindow(BoxLayout):
	def __init__(self, image_array, current_slice, num_slices, **kwargs):
		super().__init__(orientation='vertical', **kwargs)
		self.image_array = image_array
		self.current_slice = current_slice
		self.num_slices = num_slices
		self.play_event = None
		self.is_playing = False

		self.image_viewer = Image()
		self.add_widget(self.image_viewer)

		tool_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
		
		self.slice_slider = Slider(min=0, max=self.num_slices - 1, value=self.current_slice, step=1)
		self.slice_slider.image_viewer = self.image_viewer
		self.slice_slider.bind(value=self.on_slice_change)
		self.slice_slider.bind(on_touch_up=self.stop_playback)
		tool_layout.add_widget(self.slice_slider)
		
		self.play_button = Button(size_hint_x=None, width=40, text='Play')
		self.play_button.bind(on_release=self.toggle_play)
		tool_layout.add_widget(self.play_button)
		
		self.add_widget(tool_layout)
		
		self.display_image(self.image_array[self.current_slice])

	def on_slice_change(self, instance, value):
		slice_idx = int(value)
		self.current_slice = slice_idx
		self.display_image(self.image_array[slice_idx])

	def display_image(self, image_array):
		image_texture = self.create_texture(image_array)
		self.image_viewer.texture = image_texture
	
	def create_texture(self, image_array):
		if image_array.dtype != np.uint8:
			image_array = self.normalize_image(image_array)
		image_height, image_width = image_array.shape
		texture = Texture.create(size=(image_width, image_height), colorfmt='luminance')
		texture.blit_buffer(image_array.tobytes(), colorfmt='luminance', bufferfmt='ubyte')
		texture.flip_vertical()
		return texture
	
	def normalize_image(self, image_array):
		image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
		image_array = (image_array * 255).astype(np.uint8)
		return image_array

	def toggle_play(self, instance):
		if self.is_playing:
			self.stop_playback()
		else:
			self.start_playback()

	def start_playback(self):
		self.is_playing = True
		self.play_event = Clock.schedule_interval(self.play_next_slice, 0.5)

	def stop_playback(self, *args):
		self.is_playing = False
		if self.play_event:
			self.play_event.cancel()
			self.play_event = None

	def play_next_slice(self, dt):
		self.current_slice = (self.current_slice + 1) % self.num_slices
		self.slice_slider.value = self.current_slice
		self.on_slice_change(self.slice_slider, self.current_slice)

class SingleImageWindow(BoxLayout):
	def __init__(self, image_array, **kwargs):
		super().__init__(orientation='vertical', **kwargs)
		self.image_array = image_array
		#print("imge_array shape: ", image_array.shape)
		self.current_image, self.current_slice = 0, 0  # assuming single slice
		self.num_slice = 1

		self.image_viewer = Image()
		self.add_widget(self.image_viewer)

		# Load the initial image
		self.display_image(self.image_array)#[self.current_image])

		button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
		self.healthy_button = Button(text='Gesund', size_hint_x=0.5)
		self.pneumonia_button = Button(text='Pneumonie', size_hint_x=0.5)
		self.healthy_button.bind(on_release=self.on_healthy_button_click)
		self.pneumonia_button.bind(on_release=self.on_pneumonia_button_click)
		button_layout.add_widget(self.healthy_button)
		button_layout.add_widget(self.pneumonia_button)
		self.add_widget(button_layout)

	# def display_image(self, image_array):
	# 	image_texture = self.create_texture(image_array)
	# 	self.image_viewer.texture = image_texture

	# def create_texture(self, image_array):
	# 	from kivy.graphics.texture import Texture
	# 	if image_array.dtype != np.uint8:
	# 		image_array = self.normalize_image(image_array)
	# 	image_height, image_width = image_array.shape
	# 	texture = Texture.create(size=(image_width, image_height), colorfmt='luminance')
	# 	texture.blit_buffer(image_array.tobytes(), colorfmt='luminance', bufferfmt='ubyte')
	# 	texture.flip_vertical()
	# 	return texture
	
	def display_image(self, image_array, *args):
		# Check the number of dimensions to determine if it's grayscale or color
		if len(image_array[0].shape) == 2:
			# Grayscale image
			image_texture = self.create_texture(image_array[0], colorfmt='luminance')
		elif len(image_array[0].shape) == 3 and image_array[0].shape[2] in [3, 4]:
			# Color image (RGB or RGBA)
			image_texture = self.create_texture(image_array[0], colorfmt='rgb' if image_array[0].shape[2] == 3 else 'rgba')
		else:
			raise ValueError("Unsupported image array shape for display.")
		
		self.image_viewer.texture = image_texture

	def create_texture(self, image_array, colorfmt='rgb'):
		if image_array.dtype != np.uint8:
			image_array = self.normalize_image(image_array)
		
		image_height, image_width = image_array.shape[:2]
		texture = Texture.create(size=(image_width, image_height), colorfmt=colorfmt)
		texture.blit_buffer(image_array.tobytes(), colorfmt=colorfmt, bufferfmt='ubyte')
		texture.flip_vertical()
		return texture

	def normalize_image(self, image_array):
		image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
		image_array = (image_array * 255).astype(np.uint8)
		return image_array

	def on_healthy_button_click(self, instance):
		self.handle_prediction(0)  # 0 for Gesund

	def on_pneumonia_button_click(self, instance):
		self.handle_prediction(1)  # 1 for Pneumonie

	def handle_prediction(self, expected_label):
		predicted_label, result_image = self.predict(expected_label)
		if predicted_label == expected_label:
			# Nothing to do, the prediction is correct
			return
		else:
			# Show a popup with the error message
			self.show_popup(predicted_label, result_image)

	def predict(self, pred):
		global ai
		#predicted_label = 0 if pred else 1
		image = np.stack([self.image_array[0]] * 3, axis=2)
		result_image, predicted_label = ai.predict(image)
		return predicted_label, result_image

	def show_popup(self, predicted_label, result_image):
		content = BoxLayout(orientation='vertical')
		message = f'Fehler: Die Vorhersage war {predicted_label}.'
		content.add_widget(Label(text=message))
		
		ok_button = Button(text='OK', size_hint_y=None, height=40)
		ok_button.bind(on_release=self.dismiss_popup)
		content.add_widget(ok_button)
		
		self.popup = Popup(title='Vorhersagefehler',
						   content=content,
						   size_hint=(0.8, 0.4))
		self.popup.open()
		
		# Update the image with the new result image
		self.display_image([result_image])
  
	def dismiss_popup(self, instance):
		self.popup.dismiss()
	
class ViewerScreen(Screen):
	'''
	def __init__(self, **kwargs):
		super(ViewerScreen, self).__init__(**kwargs)
		self.layout = BoxLayout(orientation='vertical')

		# Toolbar layout
		self.toolbar_layout = BoxLayout(
			size_hint=(1, 0.1), orientation='horizontal')

		# Add icons to the toolbar instead of text buttons
		self.tools = [
			('CT Value', 'ions_black/icon_ct_value.png'),
			('MPR', 'ions_black/icon_mpr.png'),
			('Capture', 'ions_black/icon_capture.png'),
			('Export', 'ions_black/icon_export.png'),
			('Rotate', 'ions_black/icon_rotate.png'),
			('Flip Horizontal', 'ions_black/icon_flip_horizontal.png'),
			('Flip Vertical', 'ions_black/icon_flip_vertical.png'),
			('Annotate', 'ions_black/icon_annotate.png')
		]

		self.tool_buttons = []
		for tool, icon in self.tools:
			btn = Button(background_normal=icon, size_hint_x=None, width=50)
			btn.bind(on_release=lambda instance,
					 x=tool: self.on_tool_select(x))
			btn.bind(on_enter=lambda instance,
					 x=tool: self.show_tooltip(instance, x))
			btn.bind(on_leave=self.hide_tooltip)
			self.tool_buttons.append(btn)

		# Add New Window button to the toolbar
		new_window_button = Button(
			text='New Window', size_hint_x=None, width=50)
		new_window_button.bind(on_release=self.add_new_window)
		self.toolbar_layout.add_widget(new_window_button)

		# Add all tool buttons
		total_width = new_window_button.width
		visible_tool_buttons = []
		for btn in self.tool_buttons:
			if total_width + btn.width > self.toolbar_layout.width:
				break
			self.toolbar_layout.add_widget(btn)
			total_width += btn.width
			visible_tool_buttons.append(btn)

		# Add remaining tools in dropdown if they don't fit
		remaining_tools = len(self.tool_buttons) - len(visible_tool_buttons)
		if remaining_tools > 0:
			dropdown_button = Button(text='⋮', size_hint_x=None, width=50)
			self.dropdown = DropDown()
			for tool, icon in self.tools[-remaining_tools:]:
				btn = Button(background_normal=icon,
							 size_hint_y=None, height=44)
				btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
				self.dropdown.add_widget(btn)
			dropdown_button.bind(on_release=self.dropdown.open)
			self.toolbar_layout.add_widget(dropdown_button)

		# Add settings button to the right upper corner of the toolbar
		# size_hint=(0.2, 1))
		settings_button = Button(text='Settings', width=50)
		settings_button.bind(on_release=self.open_settings)
		self.toolbar_layout.add_widget(settings_button)

		self.layout.add_widget(self.toolbar_layout)

		# Image container with GridLayout for flexible arrangement
		self.image_container = GridLayout(cols=1, spacing=10, size_hint_y=None)
		self.image_container.bind(
			minimum_height=self.image_container.setter('height'))
		self.layout.add_widget(self.image_container)

		self.add_widget(self.layout)
		self.dicom_data = None
		self.image_array = None
		self.num_slices = 0
		self.current_slice = 0
		self.load_initial_window = True  # Flag to load initial window
	'''

	def __init__(self, **kwargs):
		super(ViewerScreen, self).__init__(**kwargs)
		self.layout = BoxLayout(orientation='vertical')

		#for the pla event:
		self.play_event = None
		self.is_playing = False

		# Toolbar layout
		self.toolbar_layout = BoxLayout(size_hint_y=None, orientation='horizontal')
		self.toolbar_layout.height = 50
		print("toolbar height:", self.toolbar_layout.height)
		# Back button
		button_size = self.toolbar_layout.height  # Adjust for padding
		self.back_button = Button(text='Back', size_hint=(None, None), size=(button_size, button_size))
		self.back_button.bind(on_release=self.go_back)
		self.toolbar_layout.add_widget(self.back_button)

		# Tools with icons
		self.tools = [
			('New Window', 'icons_black/icon_new_window.png'),
			('CT Value', 'icons_black/icon_ct_value.png'),
			('MPR', 'icons_black/icon_mpr.png'),
			('Capture', 'icons_black/icon_capture.png'),
			('Export', 'icons_black/icon_export.png'),
			('Rotate', 'icons_black/icon_rotate.png'),
			('Flip Horizontal', 'icons_black/icon_flip_horizontal.png'),
			('Flip Vertical', 'icons_black/icon_flip_vertical.png'),
			('Annotate', 'icons_black/icon_annotate.png'), 
			('AI', 'icons_black/icon_ai.png')
		]

		self.tool_buttons = []
		for tool, icon in self.tools:
			#btn = Button(background_normal=icon, background_down=icon, size_hint=(None, None), size=(button_size, button_size))
			btn = Button(text = tool, background_down=icon, size_hint=(None, None), size=(button_size+50, button_size))
			btn.bind(on_release=lambda instance, x=tool: self.on_tool_select(x))
			btn.bind(on_enter=lambda instance, x=tool: self.show_tooltip(instance, x))
			btn.bind(on_leave=self.hide_tooltip)
			self.tool_buttons.append(btn)

		# Settings button
		self.settings_button = Button(text='Settings', size_hint=(None, None), size=(button_size, button_size))
		self.settings_button.bind(on_release=self.open_settings)
		
		# Dropdown button
		self.dropdown_button = Button(text='⋮', size_hint=(None, None), size=(button_size, button_size))
		self.dropdown = DropDown()

		# Add all widgets initially
		self.layout.add_widget(self.toolbar_layout, index=0)
		self.toolbar_layout.add_widget(self.settings_button)

		# Image container with GridLayout for flexible arrangement
		self.image_container = GridLayout(cols=1, spacing=10, size_hint_y=0.8)
		self.image_container.bind(minimum_height=self.image_container.setter('height'))
		self.layout.add_widget(self.image_container)

		self.add_widget(self.layout)
		self.dicom_data = None
		self.image_array = None
		self.num_slices = 0
		self.current_slice = 0
		self.load_initial_window = True  # Flag to load initial window

		# Bind the resize event to the method
		Window.bind(on_resize=self.update_toolbar)

		# Initial toolbar update
		self.update_toolbar(Window, Window.width, Window.height)
	
	def update_toolbar(self, window, width, height):
		self.toolbar_layout.clear_widgets()
		self.dropdown.clear_widgets()

		self.toolbar_layout.add_widget(self.back_button)

		button_size = self.toolbar_layout.height  # Adjust for padding
		available_width = width - self.back_button.width - self.settings_button.width - self.dropdown_button.width

		total_width = 0
		visible_tool_buttons = []

		for btn in self.tool_buttons:
			if total_width + btn.width > available_width:
				break
			self.toolbar_layout.add_widget(btn)
			total_width += btn.width
			visible_tool_buttons.append(btn)

		# Add remaining tools in dropdown if they don't fit
		remaining_tools = len(self.tool_buttons) - len(visible_tool_buttons)
		if remaining_tools > 0:
			for tool, icon in self.tools[len(visible_tool_buttons):]:
				btn = Button(background_normal=icon, size_hint_y=None, height=button_size)
				btn.bind(on_release=lambda instance, x=tool: self.on_tool_select(x))
				self.dropdown.add_widget(btn)
			self.dropdown_button.bind(on_release=self.dropdown.open)
			self.toolbar_layout.add_widget(self.dropdown_button)

		self.toolbar_layout.add_widget(self.settings_button)

	def show_tooltip(self, instance, tool):
		self.tooltip = Tooltip(text=tool)
		self.tooltip.open(instance)

	def hide_tooltip(self, instance):
		if self.tooltip:
			self.tooltip.dismiss()

	def on_tool_select(self, tool):
		print(f"Selected tool: {tool}")
		if tool == 'Rotate':
			self.rotate_image()
		elif tool == 'Flip Horizontal':
			self.flip_image(horizontal=True)
		elif tool == 'Flip Vertical':
			self.flip_image(horizontal=False)
		elif tool == 'MPR':
			self.show_mpr()
		elif tool == 'CT Value':
			self.show_ct_value()
		elif tool == 'Capture':
			self.capture_image()
		elif tool == 'Export':
			self.export_image()
		elif tool == 'Annotate':
			self.annotate_image()
		elif tool == "New Window":
			self.add_new_window()
		elif tool == "AI":
			self.showAI()

	
	def load_dicom(self, dicom_file):
		dicom_data = pydicom.dcmread(dicom_file)
		self.image_array = dicom_data.pixel_array
		self.dicom_data = dicom_data

		slider = False
  
		self.image_container.clear_widgets()
		if len(self.image_array.shape) == 3:  # If the image is a volume
			self.num_slices = self.image_array.shape[0]
			self.current_slice = 0
			slider = True
		else:
			self.image_array = [self.image_array]
			self.num_slices = 1
			self.current_slice = 0

		# Create and add initial image window
		if self.load_initial_window:
			self.add_new_window(None, slider)

	

	def create_texture(self, image_array):
		if image_array.dtype != np.uint8:
			pass#image_array = self.normalize_image(image_array)
		image_height, image_width = image_array.shape
		texture = Texture.create(size=(image_width, image_height), colorfmt='luminance')
		texture.blit_buffer(image_array.tobytes(), colorfmt='luminance', bufferfmt='ubyte')
		texture.flip_vertical()
		return texture

	def normalize_image(self, image_array):
		image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
		image_array = (image_array * 255).astype(np.uint8)
		return image_array

	

	def display_image(self, image_array, image_viewer, zoom=1):
		if len(image_array.shape) == 2:
			height, width = image_array.shape
		else:
			raise ValueError("Expected a 2D array for display_image")

		zoomed_array = cv2.resize(
			image_array, (int(width * zoom), int(height * zoom)))
		image_texture = self.numpy_to_texture(zoomed_array)
		image_viewer.texture = image_texture

	def numpy_to_texture(self, image_array):
		image_array = np.flipud(image_array)
		buffer = image_array.tobytes()
		texture = Texture.create(
			size=(image_array.shape[1], image_array.shape[0]), colorfmt='luminance')
		texture.blit_buffer(buffer, colorfmt='luminance', bufferfmt='ubyte')
		return texture

	def on_slice_change(self, instance, value):
		slice_idx = int(value)
		image_viewer = instance.image_viewer
		self.current_slice = slice_idx
		self.display_image(self.image_array[slice_idx], image_viewer)

	def add_new_window3(self, instance=None, slider = True):
		window_layout = BoxLayout(
			orientation='vertical', size_hint_y=None, height=300)
		image_viewer = Image()
		window_layout.add_widget(image_viewer)
		if slider:
			tool_layout = BoxLayout(
			orientation='horizontal', size_hint_y=None, height=40)
			slice_slider = Slider(min=0, max=self.num_slices - 1,
								value=self.current_slice, step=1)
			slice_slider.image_viewer = image_viewer  # Bind image_viewer to the slider
			slice_slider.bind(value=self.on_slice_change)
			tool_layout.add_widget(slice_slider)
			play_button = Button(size_hint_y=None, width = 40)
			play_button.height = slice_slider.height
			#play_button.bind(value = self.play)
			tool_layout.add_widget(play_button)
			window_layout.add_widget(tool_layout)
		self.image_container.add_widget(window_layout)
		if slider: self.display_image(self.image_array[self.current_slice], image_viewer)
		else: self.display_image(self.image_array, image_viewer)
		self.image_container.cols = int(
			len(self.image_container.children) ** 0.5)

	def add_new_window2(self, instance=None, slider=True):
		window_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=300)
		
		# Image viewer
		image_viewer = Image()
		window_layout.add_widget(image_viewer)
		
		if slider:
			# Container for slider and button
			tool_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
			
			# Slider
			self.slice_slider = Slider(min=0, max=self.num_slices - 1, value=self.current_slice, step=1)
			self.slice_slider.image_viewer = image_viewer  # Bind image_viewer to the slider
			self.slice_slider.bind(value=self.on_slice_change, on_touch_up=self.stop_playback)
			tool_layout.add_widget(self.slice_slider)
			
			# Play button
			play_button = Button(size_hint_x=None, width=40, text='Play')
			play_button.bind(on_release=self.toggle_play)
			tool_layout.add_widget(play_button)
			
			window_layout.add_widget(tool_layout)
		
		# Add window_layout to image_container
		self.image_container.add_widget(window_layout)
		
		# Display the initial image
		if slider:
			self.display_image(self.image_array[self.current_slice], image_viewer)
		else:
			self.display_image(self.image_array, image_viewer)
		
		# Update the columns in the GridLayout to fit the maximum number of windows
		self.image_container.cols = int(len(self.image_container.children) ** 0.5)

	def add_new_window(self, instance=None, slider=True):
		if self.num_slices == 1:
			image_window = SingleImageWindow(self.image_array)
		else:
			image_window = ImageWindow(self.image_array, self.current_slice, self.num_slices)
		self.image_container.add_widget(image_window)
		self.image_container.cols = int(len(self.image_container.children) ** 0.5)

	### Play button for image window
	def toggle_play(self, instance):
		if self.is_playing:
			self.stop_playback()
		else:
			self.start_playback()

	def start_playback(self):
		self.is_playing = True
		self.play_event = Clock.schedule_interval(self.play_next_slice, 0.5)

	def stop_playback(self, *args):
		self.is_playing = False
		if self.play_event:
			self.play_event.cancel()
			self.play_event = None

	def play_next_slice(self, dt):
		self.current_slice = (self.current_slice + 1) % self.num_slices
		self.slice_slider.value = self.current_slice
		self.on_slice_change(self.slice_slider, self.current_slice)

 
 
	#============= Tools =========================
	def rotate_image(self):
		
		for child in self.image_container.children:
			# child.children[0] is the Image widget
			child.image_array = np.rot90(self.image_array, axes=(1, 2))
			child.display_image(
				child.image_array[child.current_slice], child.children[0])

	def flip_image(self, horizontal=True):
		if horizontal:
			self.image_array = np.flip(self.image_array, axis=2)
		else:
			self.image_array = np.flip(self.image_array, axis=1)
		for child in self.image_container.children:
			# child.children[0] is the Image widget
			self.display_image(
				self.image_array[self.current_slice], child.children[0])

	def show_mpr(self):
		self.mpr_layout = BoxLayout(orientation='horizontal')

		self.axial_view = Image()
		self.coronal_view = Image()
		self.sagittal_view = Image()

		self.mpr_layout.add_widget(self.axial_view)
		self.mpr_layout.add_widget(self.coronal_view)
		self.mpr_layout.add_widget(self.sagittal_view)

		self.add_widget(self.mpr_layout)

		self.update_mpr_views()

	def update_mpr_views(self):
		axial_image = self.image_array[:, :, self.image_array.shape[2] // 2]
		coronal_image = self.image_array[:, self.image_array.shape[1] // 2, :]
		sagittal_image = self.image_array[self.image_array.shape[0] // 2, :, :]

		self.axial_view.texture = self.numpy_to_texture(axial_image)
		self.coronal_view.texture = self.numpy_to_texture(coronal_image)
		self.sagittal_view.texture = self.numpy_to_texture(sagittal_image)

	def show_ct_value(self):
		for child in self.image_container.children:
			child.children[0].bind(on_touch_down=self.display_ct_value)

	def display_ct_value(self, instance, touch):
		if instance.collide_point(*touch.pos):
			x = int(touch.x * self.image_array.shape[2] / instance.width)
			y = int((instance.height - touch.y) *
					self.image_array.shape[1] / instance.height)
			ct_value = self.image_array[self.current_slice, y, x]
			popup = Popup(title='CT Value',
						  content=Label(text=f'CT Value: {ct_value}'),
						  size_hint=(0.3, 0.3))
			popup.open()

	def capture_image(self):
		plt.imshow(self.image_array[self.current_slice], cmap='gray')
		plt.title('Captured Image')
		plt.show()

	def export_image(self):
		export_popup = Popup(title='Export Image',
							 content=TextInput(hint_text='Enter file name'),
							 size_hint=(0.4, 0.3))
		export_popup.bind(on_dismiss=lambda x: self.save_image(
			export_popup.content.text))
		export_popup.open()

	def save_image(self, filename):
		if filename:
			cv2.imwrite(filename, self.image_array[self.current_slice])

	def annotate_image(self):
		for child in self.image_container.children:
			child.children[0].bind(on_touch_down=self.add_annotation)

	def add_annotation(self, instance, touch):
		if instance.collide_point(*touch.pos):
			x = int(touch.x * self.image_array.shape[2] / instance.width)
			y = int((instance.height - touch.y) *
					self.image_array.shape[1] / instance.height)
			self.image_array[self.current_slice] = cv2.circle(
				self.image_array[self.current_slice], (x, y), radius=10, color=(255, 255, 255), thickness=-1)
			self.display_image(self.image_array[self.current_slice], instance)

	def open_settings(self, instance):
		self.manager.current = 'settings'

	def go_back(self, instance):
		self.manager.current = 'start'
	
	def showAI(self):
		global ai
		image = self.image_array[0]
		if len(image.shape) == 2:  # Grayscale image
			image = np.stack([image] * 3, axis=2)
		image = ai.predict(image)
		for child in self.image_container.children:
			# child.children[0] is the Image widget
			self.display_image(
				self.image_array[self.current_slice], child.children[0])



# class ViewerScreen(Screen):
#	 def __init__(self, **kwargs):
#		 super(ViewerScreen, self).__init__(**kwargs)
#		 self.layout = BoxLayout(orientation='vertical')

#		 # Toolbar layout
#		 self.toolbar_layout = BoxLayout(size_hint_y=None, orientation='horizontal')
#		 self.toolbar_layout.height = 50

#		 # Back button
#		 button_size = self.toolbar_layout.height  # Adjust for padding
#		 self.back_button = Button(text='Back', size_hint=(None, None), size=(button_size, button_size))
#		 self.back_button.bind(on_release=self.go_back)
#		 self.toolbar_layout.add_widget(self.back_button)

#		 # Tools with icons
#		 self.tools = [
#			 ('New Window', 'icons_black/icon_new_window.png'),
#			 ('CT Value', 'icons_black/icon_ct_value.png'),
#			 ('MPR', 'icons_black/icon_mpr.png'),
#			 ('Capture', 'icons_black/icon_capture.png'),
#			 ('Export', 'icons_black/icon_export.png'),
#			 ('Rotate', 'icons_black/icon_rotate.png'),
#			 ('Flip Horizontal', 'icons_black/icon_flip_horizontal.png'),
#			 ('Flip Vertical', 'icons_black/icon_flip_vertical.png'),
#			 ('Annotate', 'icons_black/icon_annotate.png'),
#			 ('AI', 'icons_black/icon_ai.png')
#		 ]

#		 self.tool_buttons = []
#		 for tool, icon in self.tools:
#			 btn = Button(background_normal=icon, background_down=icon, size_hint=(None, None), size=(button_size, button_size))
#			 btn.bind(on_release=lambda instance, x=tool: self.on_tool_select(x))
#			 btn.bind(on_enter=lambda instance, x=tool: self.show_tooltip(instance, x))
#			 btn.bind(on_leave=self.hide_tooltip)
#			 self.tool_buttons.append(btn)

#		 # Settings button
#		 self.settings_button = Button(text='Settings', size_hint=(None, None), size=(button_size, button_size))
#		 self.settings_button.bind(on_release=self.open_settings)

#		 # Dropdown button
#		 self.dropdown_button = Button(text='⋮', size_hint=(None, None), size=(button_size, button_size))
#		 self.dropdown = DropDown()

#		 # Add all widgets initially
#		 self.layout.add_widget(self.toolbar_layout, index=0)
#		 self.toolbar_layout.add_widget(self.settings_button)

#		 # Image container with GridLayout for flexible arrangement
#		 self.image_container = GridLayout(cols=1, spacing=10, size_hint_y=0.8)
#		 self.image_container.bind(minimum_height=self.image_container.setter('height'))
#		 self.layout.add_widget(self.image_container)

#		 self.add_widget(self.layout)
#		 self.dicom_data = None
#		 self.image_array = None
#		 self.num_slices = 0
#		 self.current_slice = 0
#		 self.load_initial_window = True  # Flag to load initial window

#		 # Bind the resize event to the method
#		 Window.bind(on_resize=self.update_toolbar)

#		 # Initial toolbar update
#		 self.update_toolbar(Window, Window.width, Window.height)

#	 def update_toolbar(self, window, width, height):
#		 self.toolbar_layout.clear_widgets()
#		 self.dropdown.clear_widgets()

#		 self.toolbar_layout.add_widget(self.back_button)

#		 button_size = self.toolbar_layout.height  # Adjust for padding
#		 available_width = width - self.back_button.width - self.settings_button.width - self.dropdown_button.width

#		 total_width = 0
#		 visible_tool_buttons = []

#		 for btn in self.tool_buttons:
#			 if total_width + btn.width > available_width:
#				 break
#			 self.toolbar_layout.add_widget(btn)
#			 total_width += btn.width
#			 visible_tool_buttons.append(btn)

#		 # Add remaining tools in dropdown if they don't fit
#		 remaining_tools = len(self.tool_buttons) - len(visible_tool_buttons)
#		 if remaining_tools > 0:
#			 for tool, icon in self.tools[len(visible_tool_buttons):]:
#				 btn = Button(background_normal=icon, size_hint_y=None, height=button_size)
#				 btn.bind(on_release=lambda instance, x=tool: self.on_tool_select(x))
#				 self.dropdown.add_widget(btn)
#			 self.dropdown_button.bind(on_release=self.dropdown.open)
#			 self.toolbar_layout.add_widget(self.dropdown_button)

#		 self.toolbar_layout.add_widget(self.settings_button)

#	 def show_tooltip(self, instance, tool):
#		 self.tooltip = Tooltip(text=tool)
#		 self.tooltip.open(instance)

#	 def hide_tooltip(self, instance):
#		 if self.tooltip:
#			 self.tooltip.dismiss()

#	 def on_tool_select(self, tool):
#		 print(f"Selected tool: {tool}")
#		 if tool == 'Rotate':
#			 self.rotate_image()
#		 elif tool == 'Flip Horizontal':
#			 self.flip_image(horizontal=True)
#		 elif tool == 'Flip Vertical':
#			 self.flip_image(horizontal=False)
#		 elif tool == 'MPR':
#			 self.show_mpr()
#		 elif tool == 'CT Value':
#			 self.show_ct_value()
#		 elif tool == 'Capture':
#			 self.capture_image()
#		 elif tool == 'Export':
#			 self.export_image()
#		 elif tool == 'Annotate':
#			 self.annotate_image()
#		 elif tool == "New Window":
#			 self.add_new_window()
#		 elif tool == "AI":
#			 self.showAI()

#	 def load_dicom(self, dicom_file):
#		 dicom_data = pydicom.dcmread(dicom_file)
#		 self.image_array = dicom_data.pixel_array
#		 self.dicom_data = dicom_data

#		 self.image_container.clear_widgets()
#		 if len(self.image_array.shape) == 3:  # If the image is a volume
#			 self.num_slices = self.image_array.shape[0]
#			 self.current_slice = self.num_slices // 2
#			 for i in range(min(self.num_slices, 1)):
#				 slice_img = self.image_array[i]
#				 self.add_image(slice_img, i)
#			 self.add_slider()  # Add slider for changing slices

#		 else:
#			 self.num_slices = 1
#			 self.current_slice = 0
#			 self.add_image(self.image_array)

#	 def add_slider(self):
#		 self.slider = Slider(min=0, max=self.num_slices - 1, value=self.current_slice, size_hint_y=None, height=50)
#		 self.slider.bind(value=self.on_slider_value_change)
#		 self.layout.add_widget(self.slider)

#	 def on_slider_value_change(self, instance, value):
#		 slice_idx = int(value)
#		 self.current_slice = slice_idx
#		 self.image_container.clear_widgets()
#		 self.add_image(self.image_array[slice_idx], slice_idx)

#	 def add_image(self, image_array, slice_idx=0):
#		 image_texture = self.create_texture(image_array)
#		 dicom_image = Image(texture=image_texture)
#		 dicom_image.slice_idx = slice_idx  # Store slice index for reference
#		 dicom_image.bind(on_touch_down=self.on_image_click)
#		 self.image_container.add_widget(dicom_image)

#	 def create_texture(self, image_array):
#		 if image_array.dtype != np.uint8:
#			 image_array = self.normalize_image(image_array)
#		 image_height, image_width = image_array.shape
#		 texture = Texture.create(size=(image_width, image_height), colorfmt='luminance')
#		 texture.blit_buffer(image_array.tobytes(), colorfmt='luminance', bufferfmt='ubyte')
#		 texture.flip_vertical()
#		 return texture

#	 def normalize_image(self, image_array):
#		 image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
#		 image_array = (image_array * 255).astype(np.uint8)
#		 return image_array

#	 def rotate_image(self):
#		 if self.image_array is not None:
#			 self.image_array = np.rot90(self.image_array)
#			 self.load_dicom(self.dicom_data)

#	 def flip_image(self, horizontal=True):
#		 if self.image_array is not None:
#			 if horizontal:
#				 self.image_array = np.flip(self.image_array, axis=2)
#			 else:
#				 self.image_array = np.flip(self.image_array, axis=1)
#			 self.load_dicom(self.dicom_data)

#	 def open_settings(self, instance):
#		 settings_popup = Popup(title='Settings', size_hint=(0.5, 0.5))
#		 settings_layout = BoxLayout(orientation='vertical')
#		 settings_layout.add_widget(Label(text='Settings will be here'))
#		 close_button = Button(text='Close')
#		 close_button.bind(on_release=settings_popup.dismiss)
#		 settings_layout.add_widget(close_button)
#		 settings_popup.add_widget(settings_layout)
#		 settings_popup.open()

#	 def go_back(self, instance):
#		 self.manager.current = 'start'

#	 def on_image_click(self, instance, touch):
#		 if instance.collide_point(*touch.pos):
#			 self.show_ct_value(instance, touch)

#	 def show_ct_value(self, instance, touch):
#		 x, y = int(touch.x), int(touch.y)
#		 if len(self.image_array.shape) == 3:
#			 value = self.image_array[self.current_slice, y, x]
#		 else:
#			 value = self.image_array[y, x]
#		 print(f"CT value at ({x}, {y}): {value}")

#	 def show_mpr(self):
#		 if len(self.image_array.shape) == 3:
#			 sagittal = self.image_array[:, :, self.image_array.shape[2] // 2]
#			 coronal = self.image_array[:, self.image_array.shape[1] // 2, :]
#			 self.show_slice_image(sagittal, title='Sagittal View')
#			 self.show_slice_image(coronal, title='Coronal View')
#		 else:
#			 print("MPR requires a 3D volume.")

#	 def show_slice_image(self, image_array, title='Slice Image'):
#		 plt.imshow(image_array, cmap='gray')
#		 plt.title(title)
#		 plt.show()

#	 def capture_image(self):
#		 if self.image_array is not None:
#			 output_path = 'captured_image.png'
#			 plt.imsave(output_path, self.image_array[self.current_slice], cmap='gray')
#			 print(f"Captured image saved as {output_path}")

#	 def export_image(self):
#		 if self.image_array is not None:
#			 output_path = 'exported_image.png'
#			 plt.imsave(output_path, self.image_array[self.current_slice], cmap='gray')
#			 print(f"Exported image saved as {output_path}")

#	 def annotate_image(self):
#		 print("Annotation functionality will be implemented here.")

#	 def add_new_window(self):
#		 new_window = ViewerScreen(name=f'viewer_{len(self.manager.screens) + 1}')
#		 self.manager.add_widget(new_window)
#		 self.manager.current = new_window.name

#	 def showAI(self):
#		 print("AI functionality will be implemented here.")


class SettingsScreen(Screen):
	def _init_(self, **kwargs):
		super(SettingsScreen, self)._init_(**kwargs)
		layout = BoxLayout(orientation='vertical')

		label = Label(text="Settings")
		layout.add_widget(label)

		back_button = Button(text='Back')
		back_button.bind(on_release=self.go_back)
		layout.add_widget(back_button)

		self.add_widget(layout)

	def go_back(self, instance):
		self.manager.current = 'start'


class DicomViewerApp(App):
	def build(self):
		sm = ScreenManager()
		sm.add_widget(StartScreen(name='start'))
		sm.add_widget(ViewerScreen(name='viewer'))
		sm.add_widget(SettingsScreen(name='settings'))
		return sm


# classe for the model
class MRCNN():
	def __init__(self, model_path, num_classes=2):
		self.num_classes = num_classes
		self.device = torch.device(
			'cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.model = self.get_model_instance_segmentation()
		self.model.load_state_dict(torch.load(model_path, map_location=self.device))
		self.model.to(self.device)
		self.model.eval()
		#return self.model, self.device

	def get_model_instance_segmentation(self):
		self.model = maskrcnn_resnet50_fpn(pretrained=True)
		in_features = self.model.roi_heads.box_predictor.cls_score.in_features
		self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
			in_features, self.num_classes)
		self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
			self.model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, self.num_classes)
		return self.model

	def predict2(self, image):
		self.model.eval()
		with torch.no_grad():
			image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
			outputs = self.model(image_tensor)

		output = outputs[0]
		masks = output['masks'].cpu().numpy()
		scores = output['scores'].cpu().numpy()
		labels = output['labels'].cpu().numpy()

		# Process image for display
		image = (image * 255).astype(np.uint8)

		for mask, label, score in zip(masks, labels, scores):
			if score < 0.5:  # Skip detections with low confidence
				continue
			mask = mask[0]
			colored_mask = np.zeros(
				(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
			colored_mask[mask > 0.5] = [255, 0, 0]  # Red color

			# Overlay mask
			blended_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

			# Draw mask boundary
			contours, _ = cv2.findContours((mask > 0.5).astype(
				np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			if contours:  # Check if contours are found
				cv2.drawContours(blended_image, contours, -1,
								 (255, 0, 0), 2)  # Red color for boundary

				# Add score text
				x, y, w, h = cv2.boundingRect(contours[0])
				cv2.putText(blended_image, f'Score: {score:.2f}', (
					x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				cv2.putText(blended_image, f'Class: {label}', (
					x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		return blended_image

	def predict(self, image):
		self.model.eval()
		with torch.no_grad():
			image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
			outputs = self.model(image_tensor)

		output = outputs[0]
		masks = output['masks'].cpu().numpy()
		scores = output['scores'].cpu().numpy()
		labels = output['labels'].cpu().numpy()

		# Process image for display
		image = (image * 255).astype(np.uint8)
		
		pneumonia_detected = False
		blended_image = image.copy()

		for mask, label, score in zip(masks, labels, scores):
			if score < 0.3:  # Skip detections with low confidence
				continue
			if label == 1:  # Assuming label 1 corresponds to Pneumonie
				pneumonia_detected = True
				mask = mask[0]
				colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
				colored_mask[mask > 0.5] = [255, 0, 0]  # Red color

				# Overlay mask
				blended_image = cv2.addWeighted(image, 0.8*score, colored_mask, 0.3, 0)

				# Draw mask boundary
				contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				if contours:  # Check if contours are found
					cv2.drawContours(blended_image, contours, -1, (255, 0, 0), 2)  # Red color for boundary

					# Add score text
					x, y, w, h = cv2.boundingRect(contours[0])
					cv2.putText(blended_image, f'Score: {score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
					cv2.putText(blended_image, f'Class: {label}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		# Return the label and the processed image
		label = 1 if pneumonia_detected else 0
		return blended_image, label

'''
def main():
	image_dir = 'stage_2_train_images'
	label_file = 'stage_2_train_labels.csv'
	val_file = 'val.csv'
	# Pfad zur vortrainierten Modell-Datei
	model_path = 'model_epoch_6_num_300_loss104.5621.pth'

	# Datensatz in train und val zerteilen:
	# CSV-Datei einlesen
	df = pd.read_csv(label_file)

	# Zeilen mischen
	df = df.sample(frac=1)

	# Daten in Trainings- und Validierungssets aufteilen
	_, val_df = train_test_split(df, test_size=0.2)

	# Validierungsdaten in CSV-Datei speichern
	val_df.to_csv(val_file, index=False)

	model, device = init_model(model_path, num_classes=2)

	# Load validation dataset
	dataset_val = PneumoniaDataset(image_dir, val_file)
	dataloader_val = DataLoader(
		dataset_val, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

	# Process and save images with masks
	print("Processing and saving images with masks...")
	output_dir = 'output_images'
	num_images_to_process = 10  # Specify the number of images to process
	save_images_with_masks(model, dataloader_val, device,
						   num_images_to_process, output_dir)
'''


if __name__ == '__main__':

	#main()
	model_path = 'model_epoch_6_num_300_loss104.5621.pth'
	ai = MRCNN(model_path, num_classes=2)
	


	DicomViewerApp().run()
