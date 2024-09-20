import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics import Color, Rectangle
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget

import vtk
import SimpleITK as sitk

import pydicom

import os

# Placeholder image loader for simulation
def load_dicom_image(file_path):
    return 'chest-x-ray.png'  # Placeholder for actual image loading



class VTKWidget(Widget):
    def __init__(self, dicom_directory, **kwargs):
        super().__init__(**kwargs)

        # Create a renderer, render window, and interactor
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_interactor = vtk.vtkRenderWindowInteractor()
        self.render_interactor.SetRenderWindow(self.render_window)

        # Load DICOM data
        self.load_dicom_data(dicom_directory)

        # Camera setup for manipulation
        self.camera = self.renderer.GetActiveCamera()

        # Create sliders for interactivity
        self.create_sliders()

        # Set background color for the renderer
        self.renderer.SetBackground(0.1, 0.2, 0.3)

        # Setup the render window size
        self.render_window.SetSize(600, 600)

        # Start the interaction
        self.render_window.Render()
        self.render_interactor.Start()
    
    def load_dicom_data(self, dicom_directory):
        """Loads and segments DICOM data using image processing algorithms"""

        # Step 1: Load DICOM images from the directory using SimpleITK
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Step 2: Apply image processing algorithms (e.g., thresholding, smoothing, etc.)
        # Here we apply a simple threshold-based segmentation for bones
        lower_threshold = 300  # Adjust this for specific segmentation
        upper_threshold = 1200

        segmented_image = sitk.Threshold(image, lower=lower_threshold, upper=upper_threshold, outsideValue=0)

        # Convert SimpleITK image to VTK format for visualization
        vtk_image = self.convert_sitk_to_vtk(segmented_image)

        # Set up the VTK pipeline with the segmented data
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image)

        self.setup_volume_rendering(volume_mapper)

    def convert_sitk_to_vtk(self, sitk_image):
        """Convert a SimpleITK image to a VTK image for visualization."""
        array_data = sitk.GetArrayFromImage(sitk_image)
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(array_data.shape)
        vtk_image.SetSpacing(sitk_image.GetSpacing())
        vtk_image.SetOrigin(sitk_image.GetOrigin())

        # Set pixel data
        flat_data = array_data.flatten(order='C')
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image
    
    def show_dicom_metadata(self, dicom_directory):
        """Load and display DICOM metadata from a directory"""
        # Load the first DICOM file to get metadata
        dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
        first_dicom_file = os.path.join(dicom_directory, dicom_files[0])

        dicom_data = pydicom.dcmread(first_dicom_file)

        # Extract and display relevant metadata
        patient_name = dicom_data.PatientName
        study_date = dicom_data.StudyDate
        modality = dicom_data.Modality

        print(f"Patient Name: {patient_name}")
        print(f"Study Date: {study_date}")
        print(f"Modality: {modality}")

        # Show metadata in a label or popup window in the GUI
        metadata_popup = Popup(title="DICOM Metadata",
                            content=Label(text=f"Patient Name: {patient_name}\n"
                                                f"Study Date: {study_date}\n"
                                                f"Modality: {modality}"),
                            size_hint=(0.5, 0.5))
        metadata_popup.open()

    def load_dicom_based_segmentations(self, dicom_directory):
        """Load segmentation presets based on DICOM attributes"""
        dicom_files = [f for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
        first_dicom_file = os.path.join(dicom_directory, dicom_files[0])

        dicom_data = pydicom.dcmread(first_dicom_file)

        if dicom_data.Modality == "CT":
            # Set segmentation parameters for CT scans
            self.update_isosurface(None, 1000)
        elif dicom_data.Modality == "MR":
            # Set segmentation parameters for MR scans
            self.update_isosurface(None, 500)
        
    def load_dicom_data2(self, dicom_directory):
        """Loads DICOM data from a directory and prepares it for volumetric rendering"""

        # Step 1: Load DICOM images from a directory using VTK's DICOM reader
        dicom_reader = vtk.vtkDICOMImageReader()
        dicom_reader.SetDirectoryName(dicom_directory)
        dicom_reader.Update()

        # Step 2: Create a volume mapper and connect it to the reader
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputConnection(dicom_reader.GetOutputPort())

        # Step 3: Define the volume properties (opacity, color, etc.)
        self.volume_property = vtk.vtkVolumeProperty()

        # Create an opacity transfer function for the volume (Segmented based on anatomy)
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)
        opacity_transfer_function.AddPoint(300, 0.1)   # Soft tissue
        opacity_transfer_function.AddPoint(500, 0.3)   # Muscles
        opacity_transfer_function.AddPoint(1000, 0.7)  # Bones
        opacity_transfer_function.AddPoint(1150, 1.0)  # Very dense structures

        # Create a color transfer function for segmentation
        color_transfer_function = vtk.vtkColorTransferFunction()
        color_transfer_function.AddRGBPoint(0, 0.0, 0.0, 0.0)      # Background
        color_transfer_function.AddRGBPoint(300, 0.8, 0.3, 0.3)    # Soft tissue (red-ish)
        color_transfer_function.AddRGBPoint(500, 0.3, 0.8, 0.3)    # Muscle (green-ish)
        color_transfer_function.AddRGBPoint(1000, 0.3, 0.3, 0.8)   # Bone (blue-ish)
        color_transfer_function.AddRGBPoint(1150, 1.0, 1.0, 0.9)   # Dense structures (light yellow)

        # Set the volume properties
        self.volume_property.SetColor(color_transfer_function)
        self.volume_property.SetScalarOpacity(opacity_transfer_function)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()

        # Step 4: Create the volume actor and set its mapper and properties
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(self.volume_property)

        # Step 5: Add the volume to the renderer
        self.renderer.AddVolume(volume)
        self.renderer.ResetCamera()


    def create_sliders(self):
        """Create sliders for adjusting the isosurface, camera zoom, lighting, and shadow"""
        
        # Isosurface slider (threshold for opacity transfer function)
        self.isosurface_slider = Slider(min=0, max=2000, value=500, size_hint=(1, 0.1))
        self.isosurface_slider.bind(value=self.update_isosurface)

        # Camera zoom slider
        self.camera_zoom_slider = Slider(min=1, max=10, value=1, size_hint=(1, 0.1))
        self.camera_zoom_slider.bind(value=self.update_camera_zoom)

        # Lighting intensity slider
        self.lighting_slider = Slider(min=0, max=5, value=1, size_hint=(1, 0.1))
        self.lighting_slider.bind(value=self.update_lighting)

        # Shadow toggle button
        self.shadow_toggle = ToggleButton(text="Shadows On/Off", size_hint=(1, 0.1))
        self.shadow_toggle.bind(on_press=self.toggle_shadows)

        # Slice slider for cutting planes
        self.slice_slider = Slider(min=0, max=100, value=50, size_hint=(1, 0.1))
        self.slice_slider.bind(value=self.update_slice)

        # Adding sliders to the layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Label(text="Isosurface Value"))
        layout.add_widget(self.isosurface_slider)
        layout.add_widget(Label(text="Camera Zoom"))
        layout.add_widget(self.camera_zoom_slider)
        layout.add_widget(Label(text="Lighting Intensity"))
        layout.add_widget(self.lighting_slider)
        layout.add_widget(Label(text="Shadow Toggle"))
        layout.add_widget(self.shadow_toggle)
        layout.add_widget(Label(text="Slice Plane"))
        layout.add_widget(self.slice_slider)

        self.add_widget(layout)

    def update_lighting(self, instance, value):
        """Update the lighting intensity of the volume rendering."""
        light = self.renderer.GetLights().GetItemAsObject(0)
        light.SetIntensity(value)
        self.render_window.Render()

    def toggle_shadows(self, instance):
        """Toggle shadows on/off for the volume."""
        if self.shadow_toggle.state == 'down':
            self.volume_property.ShadeOn()
        else:
            self.volume_property.ShadeOff()
        self.render_window.Render()

    def update_slice(self, instance, value):
        """Update the slice plane to view specific parts of the Ends with a cutting plane."""
        self.volume_mapper.AddClippingPlane(vtk.vtkPlane())
        clipping_plane = self.volume_mapper.GetClippingPlane(0)
        clipping_plane.SetOrigin(value, 0, 0)
        clipping_plane.SetNormal(1, 0, 0)
        self.render_window.Render()

    def update_isosurface(self, instance, value):
        """Adjust the opacity transfer function based on the isosurface value"""
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        opacity_transfer_function.AddPoint(0, 0.0)
        opacity_transfer_function.AddPoint(value, 0.2)  # Adjust based on slider
        opacity_transfer_function.AddPoint(1000, 0.85)
        opacity_transfer_function.AddPoint(1150, 1.0)

        self.volume_property.SetScalarOpacity(opacity_transfer_function)
        self.render_window.Render()

    def update_camera_zoom(self, instance, value):
        """Zoom in/out on the 3D volume by adjusting the camera's distance"""
        self.camera.SetDistance(value * 100)
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()




# A widget to wrap the image and allow interaction (Zoom, Pan)
class ImageDisplayWidget(Scatter):
    def __init__(self, img_src, **kwargs):
        super().__init__(**kwargs)
        self.do_rotation = False  # Rotation ist deaktiviert
        self.do_translation = True
        self.do_scale = True
        
        self.image = Image(source=img_src)
        self.add_widget(self.image)
        
        # Neue Eigenschaften für Anmerkungen
        self.annotation_mode = False
        self.line = None

    def toggle_annotation_mode(self):
        """Schaltet den Anmerkungsmodus um (zum Zeichnen von Linien)"""
        self.annotation_mode = not self.annotation_mode

    def on_touch_down(self, touch):
        if self.annotation_mode:
            with self.canvas:
                Color(1, 0, 0)  # Rote Farbe für Messungen
                self.line = Line(points=(touch.x, touch.y), width=2)
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.annotation_mode and self.line:
            self.line.points += [touch.x, touch.y]
        return super().on_touch_move(touch)



# Main Application
class DicomViewerApp(App):
    
    '''
    def build(self):
            
        # Main layout (GridLayout with toolbar and main area)
        main_layout = BoxLayout(orientation='vertical')


        # Top toolbar
        self.toolbar = BoxLayout(size_hint_y=0.1, padding=5, spacing=5)
        tool_buttons = ['Zoom', 'Pan', 'Rotate', 'Window/Level', 'Measure']
        for tool in tool_buttons:
            btn = Button(text=tool, size_hint_x=None, width=100)
            btn.bind(on_press=self.apply_tool)
            self.toolbar.add_widget(btn)
        
        # Button zum Aktivieren des Anmerkungsmodus in der Toolbar
        annotation_btn = Button(text='Annotation Mode', size_hint_x=None, width=120)
        annotation_btn.bind(on_press=self.toggle_annotation_mode)
        self.toolbar.add_widget(annotation_btn)

        
        # Hinzugefügte Toolbar-Buttons für Layout-Wechsel
        layout_options = ['2x2 Layout', '3x3 Layout']
        for layout in layout_options:
            layout_btn = Button(text=layout, size_hint_x=None, width=100)
            layout_btn.bind(on_press=self.change_layout)
            self.toolbar.add_widget(layout_btn)


        # Main image viewing area
        self.image_area = BoxLayout(orientation='horizontal', padding=10)


        # Left sidebar for file navigation
        file_sidebar = ScrollView(size_hint_x=0.2)
        file_list = GridLayout(cols=1, spacing=10, size_hint_y=None)
        file_list.bind(minimum_height=file_list.setter('height'))


        # Adding sample DICOM files (this should be dynamically loaded in a real application)
        for i in range(10):
            file_btn = Button(text=f'Patient {i+1}', size_hint_y=None, height=50)
            file_btn.bind(on_press=self.load_image_series)
            file_list.add_widget(file_btn)

        file_sidebar.add_widget(file_list)
        self.image_area.add_widget(file_sidebar)


        # Image viewing window with a scrollbar
        image_display_area = BoxLayout(orientation='vertical')
        image_scroll = ScrollView(size_hint_y=0.9)

        self.image_grid = GridLayout(cols=1, spacing=10, size_hint_y=None)
        self.image_grid.bind(minimum_height=self.image_grid.setter('height'))

        # Sample image placeholders with selectable functionality
        for i in range(5):
            img = ImageDisplayWidget(img_src='chest-x-ray.png', size_hint_y=None, height=200)
            img.bind(on_touch_down=self.set_active_window)
            self.image_grid.add_widget(img)

        image_scroll.add_widget(self.image_grid)


        # Buttons for scrolling through images
        image_controls = BoxLayout(size_hint_y=0.1)
        btn_previous = Button(text='Previous')
        btn_next = Button(text='Next')

        image_controls.add_widget(btn_previous)
        image_controls.add_widget(btn_next)

        image_display_area.add_widget(image_scroll)
        image_display_area.add_widget(image_controls)


        # Adding the image display area to the main layout
        self.image_area.add_widget(image_display_area)
        
        
        # Adding main areas to the window
        main_layout.add_widget(self.toolbar)
        main_layout.add_widget(self.image_area)


        # Placeholder for active window reference
        self.active_window = None
        self.active_border_color = [1, 0, 0, 1]  # Red color for active window border


        # Hinzufügen eines Sliders für Window/Level-Einstellungen
        window_slider = Slider(min=0, max=255, value=128)
        level_slider = Slider(min=0, max=255, value=128)

        window_slider.bind(value=self.update_window_level)
        level_slider.bind(value=self.update_window_level)

        self.toolbar.add_widget(Label(text='Window'))
        self.toolbar.add_widget(window_slider)
        self.toolbar.add_widget(Label(text='Level'))
        self.toolbar.add_widget(level_slider)
        
        
        # 3D Model Button
        btn_3d = Button(text='Show 3D CT/MRI', size_hint_x=None, width=120)
        btn_3d.bind(on_press=self.show_3d_model)
        self.toolbar.add_widget(btn_3d)
        
        return main_layout
    '''
    
    def build(self):
        # Main layout (GridLayout with toolbar and main area)
        main_layout = BoxLayout(orientation='vertical')

        # Top toolbar
        self.toolbar = BoxLayout(size_hint_y=0.1, padding=5, spacing=5)
        tool_buttons = ['Zoom', 'Pan', 'Rotate', 'Window/Level', 'Measure']
        for tool in tool_buttons:
            btn = Button(text=tool, size_hint_x=None, width=100)
            btn.bind(on_press=self.apply_tool)
            self.toolbar.add_widget(btn)
        
        # Annotation mode button in toolbar
        annotation_btn = Button(text='Annotation Mode', size_hint_x=None, width=120)
        annotation_btn.bind(on_press=self.toggle_annotation_mode)
        self.toolbar.add_widget(annotation_btn)

        # Layout buttons (2x2 and 3x3)
        layout_options = ['2x2 Layout', '3x3 Layout']
        for layout in layout_options:
            layout_btn = Button(text=layout, size_hint_x=None, width=100)
            layout_btn.bind(on_press=self.change_layout)
            self.toolbar.add_widget(layout_btn)

        # 3D Model Button
        btn_3d = Button(text='Show 3D CT/MRI', size_hint_x=None, width=120)
        btn_3d.bind(on_press=self.show_3d_model)
        self.toolbar.add_widget(btn_3d)

        # Window/Level sliders
        window_slider = Slider(min=0, max=255, value=128)
        level_slider = Slider(min=0, max=255, value=128)
        window_slider.bind(value=self.update_window_level)
        level_slider.bind(value=self.update_window_level)
        self.toolbar.add_widget(Label(text='Window'))
        self.toolbar.add_widget(window_slider)
        self.toolbar.add_widget(Label(text='Level'))
        self.toolbar.add_widget(level_slider)

        # Main image viewing area
        self.image_area = BoxLayout(orientation='horizontal', padding=10)

        # Left sidebar for file navigation
        left_sidebar = BoxLayout(orientation='vertical', size_hint_x=0.25)

        # File chooser for directory and file navigation
        self.filechooser = FileChooserListView(path=os.getcwd(), filters=['*.dcm'])
        self.filechooser.bind(on_selection=self.load_image_series)
        left_sidebar.add_widget(self.filechooser)

        # Button to toggle between file explorer and patient info
        toggle_view_btn = Button(text="Toggle Patient Info", size_hint_y=None, height=40)
        toggle_view_btn.bind(on_press=self.toggle_patient_info)
        left_sidebar.add_widget(toggle_view_btn)

        # Patient information area (optional view)
        self.patient_info_label = Label(text='No Patient Selected', size_hint_y=None, height=200)
        left_sidebar.add_widget(self.patient_info_label)

        # Add the sidebar to the main layout
        self.image_area.add_widget(left_sidebar)

        # Image viewing window with a scrollbar
        image_display_area = BoxLayout(orientation='vertical')
        image_scroll = ScrollView(size_hint_y=0.9)

        self.image_grid = GridLayout(cols=2, spacing=10, size_hint_y=None)
        self.image_grid.bind(minimum_height=self.image_grid.setter('height'))

        # Placeholder images with selectable functionality (dynamically loaded)
        image_scroll.add_widget(self.image_grid)
        image_display_area.add_widget(image_scroll)

        # Image control buttons for navigating through images
        image_controls = BoxLayout(size_hint_y=0.1)
        btn_previous = Button(text='Previous')
        btn_next = Button(text='Next')
        image_controls.add_widget(btn_previous)
        image_controls.add_widget(btn_next)
        image_display_area.add_widget(image_controls)

        # Add the image display area to the main layout
        self.image_area.add_widget(image_display_area)

        # Add toolbar and image area to the window
        main_layout.add_widget(self.toolbar)
        main_layout.add_widget(self.image_area)

        # Initialize variables
        self.active_window = None
        self.active_border_color = [1, 0, 0, 1]  # Red color for active window border

        return main_layout
    
    
    def toggle_patient_info(self, instance):
        """Toggle between file explorer and patient information view."""
        if self.filechooser.parent:
            self.image_area.remove_widget(self.filechooser)
            self.image_area.add_widget(self.patient_info_label)
        else:
            self.image_area.remove_widget(self.patient_info_label)
            self.image_area.add_widget(self.filechooser)

    def load_image_series(self, filechooser, selection):
        """Load DICOM image series when a file is selected."""
        if not selection:
            return

        dicom_file = selection[0]
        print(f'Loading DICOM file: {dicom_file}')

        # Update patient info
        patient_name = self.get_patient_name(dicom_file)
        self.patient_info_label.text = f'Patient: {patient_name}'

        # Simulate loading DICOM images
        self.image_grid.clear_widgets()
        for i in range(4):
            img = ImageDisplayWidget(img_src='your_image_placeholder.png', size_hint_y=None, height=400)
            img.bind(on_touch_down=self.set_active_window)
            self.image_grid.add_widget(img)

    def get_patient_name(self, dicom_file):
        """Extract patient name from DICOM file (for demonstration purposes)."""
        # In a better implementation, you'd actually read the DICOM file
        return "John Doe"  # Placeholder
    
    def update_window_level(self, instance, value):
        """Update the window/level for the active window (DICOM image adjustments)"""
        if not self.active_window:
            return
        
        window = int(value)  # Placeholder window calculation
        level = int(instance.value)  # Placeholder level calculation

        print(f"Window: {window}, Level: {level}")

        # TODO: Apply the window/level adjustments to the DICOM image display
        # The image processing logic would use these values to adjust brightness/contrast

    def toggle_annotation_mode(self, instance):
        if self.active_window:
            self.active_window.toggle_annotation_mode()
        else:
            print("Kein aktives Fenster ausgewählt.")

    
    def change_layout(self, instance):
        """ Ändert die Bildfensteranordnung (2x2 oder 3x3) """
        layout_choice = instance.text
        self.image_grid.clear_widgets()

        if layout_choice == '2x2 Layout':
            self.image_grid.cols = 2
        elif layout_choice == '3x3 Layout':
            self.image_grid.cols = 3
        
        # Dummy-Bilder in das neue Layout einfügen
        for i in range(self.image_grid.cols * self.image_grid.cols):
            img = ImageDisplayWidget(img_src='your_image_placeholder.png', size_hint_y=None, height=200)
            img.bind(on_touch_down=self.set_active_window)
            self.image_grid.add_widget(img)
    
    def show_3d_model(self, instance):
        """Load a 3D volume and display it"""
        dicom_directory = '/path/to/dicom/files'  # Adjust this to point to actual DICOM directory
        self.vtk_widget = VTKWidget(dicom_directory)
        self.vtk_widget.display_3d_model()
    
    def load_image_series(self, instance):
        # Simulating DICOM image loading logic
        print(f'Loading series for: {instance.text}')
        # Load DICOM images for this patient and update the image grid

        # Example: Update the image grid with new images
        self.image_grid.clear_widgets()
        for i in range(5):
            img = ImageDisplayWidget(img_src='your_image_placeholder.png', size_hint_y=None, height=200)
            img.bind(on_touch_down=self.set_active_window)
            self.image_grid.add_widget(img)

    def set_active_window(self, instance, touch):
        # Check if touch is within widget's boundaries and activate the window
        if instance.collide_point(*touch.pos):
            self.clear_active_window()  # Clear previous active window
            self.active_window = instance

            # Add red border to the active window
            with instance.canvas.before:
                Color(*self.active_border_color)
                instance.rect = Rectangle(size=instance.size, pos=instance.pos)
            instance.bind(pos=self.update_active_window_border, size=self.update_active_window_border)

    def update_active_window_border(self, instance, value):
        """Update the border when the window is resized or moved."""
        if instance.rect:
            instance.rect.pos = instance.pos
            instance.rect.size = instance.size

    def clear_active_window(self):
        # Remove the border from the previous active window
        if self.active_window:
            self.active_window.canvas.before.clear()
            self.active_window = None

    def apply_tool(self, instance):
        # Apply the selected tool to the active window
        if not self.active_window:
            popup = Popup(title='No Active Window',
                          content=Label(text='Please select an image window to apply the tool.'),
                          size_hint=(0.6, 0.4))
            popup.open()
            return

        tool = instance.text
        print(f'Applying {tool} to the active window')

        # Example tool application (Zoom, Pan, etc.)
        if tool == 'Zoom':
            self.active_window.scale *= 1.2  # Zoom in by increasing the scale
        elif tool == 'Pan':
            # Pan can be done by dragging, but here we show example interaction
            self.active_window.translation = (10, 10)  # Shift the image slightly
        elif tool == 'Rotate':
            self.active_window.rotation += 45  # Rotate the image by 45 degrees

if __name__ == '__main__':
    DicomViewerApp().run()
