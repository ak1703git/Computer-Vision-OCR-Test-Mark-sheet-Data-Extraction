""" Automated Exam Mark Sheets Text Extraction using OCR and Computervison Project Code"""

# Import Of Libraries
import cv2
import numpy as np
from imutils.perspective import four_point_transform
import json
import os
import google.generativeai as genai
import re
import pandas as pd
import tkinter
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk
from tkinter import Tk
import tkinter as tk
import openpyxl
from tkinter import messagebox
import PIL.Image

# GUI - Tkinter Root Window
window = tkinter.Tk()
# GUI window title
window.title("University Marksheet OCR Data Entry Application")
# GUI - Background color - HEX code
window.configure(bg="#C06C84")


# Global Variables
# Set to None untill the necessary files are loaded into the GUI using  buttons within the GUI
# University Master Data Excel Filepath
MASTER_DATA_GLOBAL_VARIABLE_FILEPATH = None
# Empty data entry excel sheet filepath
DATA_ENTRY_GLOBAL_VARIABLE_FILEPATH = None
# "Test Images" - Folder file path
IMAGES_FOLDER_PATH = None
IMAGES_LIST = [] # empty global list for images when added by the folder filepath
CURRENT_INDEX = 0 # To itirate over all images within the folder untill all images within a folder are iterated over.


# Main Definition statements of the program - Perform specific tasks.
# resizes the images in the same aspect ration with the width fixed to 500 pixels.
def resize(image, width=500):
        """this functions resizes the image for ease of use, increased speed and good processing"""
        h, w, c = image.shape # height, width & color channel holder variables
        height = int((h / w) * width) # Holds the resized hight based on the specified width
        size = width, height 
        image = cv2.resize(image, (width, height))
        return image, size

# Initiats the Document Scanner Process
def document_scanner(SOURCE_IMAGE_FILE_PATH):
    """this functions handels the document scanning function
    it takes in the original image and then processes it to
    return an image with the background removed"""
    original_image = cv2.imread(SOURCE_IMAGE_FILE_PATH) # Reads the image
    image_resized, size = resize(original_image) # resizes the image 
    kernel = np.ones((5,5),np.uint8) #creates a 5x5 matrix (kernel) of ones with a data type of np.uint8 (unsigned 8-bit integer)
    # Initiates a closing operation and does it three times.
    img = cv2.morphologyEx(image_resized, cv2.MORPH_CLOSE, kernel, iterations= 3)
    # Initializes a mask of the same height and width of the input image
    mask = np.zeros(img.shape[:2],np.uint8)
    # Creates a background model
    bgdModel = np.zeros((1,65),np.float64)
    # Creates a foreground model
    fgdModel = np.zeros((1,65),np.float64)
    # Defines a rectangle around an object
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    # Inititates application of the grapcut algorithm
    # Does 5 itirations
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    # Converts the colored image to a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Smoothens out the image suing gaussian bulr by erosion and dilation
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection. Applies the "Canny Edge Detection" method
    canny = cv2.Canny(gray, 0, 200)
    # Applying dilation by using a 5x5 structuring element
    # Dilating the image helps with better definition of shapes
    # This helps with drawing contours much better
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Extracting & Sorting contours in a desending order
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Itirating through each contour to detect a shape with four corners
    for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                four_points = np.squeeze(approx)
                break
    # Drawing A contour around the shape with four corners
    # Draws a contour around the edges of the mark sheet within the image 
    # Specifying the thickness of the contour "3"
    # Specifying the color of the contour (0, 255, 0)
    # [four_points] - Contails the 4 corner points of the contour
    # -1 draw the detected contour on the resized image  
    cv2.drawContours(image_resized, [four_points], -1, (0, 255, 0), 3)
    # Convert four_points to a NumPy Float32 Array
    four_points = np.array(four_points, dtype="float32")
    multiplier = original_image.shape[1] / size[0]
    four_points_original_image = four_points * multiplier
    four_points_original_image = four_points_original_image.astype(int)
    # Align the four points to get a vertical image
    wrap_image = four_point_transform(original_image, four_points_original_image)

    # un-commment the following comments to see intermediate images/results if necessary
    #----
    #cv2.imshow("Original Image",original_image)
    #cv2.imshow("Resized Image Image",image_resized)
    #cv2.imshow("img1",img)
    #cv2.imshow("gray", gray)
    #cv2.imshow("Edge Image", canny)
    #cv2.imshow("Contour Image", wrap_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #----
    return wrap_image

# The following functions perform an OCR task USING an API
def perform_ocr(image):
    """this functions gives us the OCR results from scanned image"""
    #Configuration - Specifying the 
    genai.configure(api_key='AIzaSyAZpIVOAW9BQLX3kx6yN6oxqzVsKpQrUcM')
    #Choose a Gemini model.
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    # Create an in-memory image object from the NumPy array
    image_pil = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Writing a prompt for gemini to extract the results into a reting as per our requirements
    prompt = """from image, get 'last name',
        'first name','student number' and questions 
        like 'Q1', 'Q2' etc and 'total' along with 
        their corresponding values within the image from 
        the entry field. return specified categories and its values only. 
        If any of the specified entries is empty then fill that space with 'EMPTY' 
        string 
        """
    # Variable that holds the respoinse from the API
    response = model.generate_content([prompt, image_pil])
    # Holdes the respinse in the form of a text - text response
    ocr_extracted_value = response.text
    #print(ocr_extracted_value)
    # Define a pattern to match key-value pairs
    pattern = r'"([^"]+)": ?"([^"]+)"'
    # Find all matches in the response text
    matches = re.findall(pattern, ocr_extracted_value)
    # Extract key-value pairs
    ocr_result_list = [(match[0], match[1]) for match in matches]
    #print(ocr_result_list)
    #print(ocr_result_list)

    # Convert this result list to a dictionary
    ocr_result_dictionary = {}
    for value_pair in ocr_result_list:
        key_value = value_pair[0]
        value = value_pair[1]
        if key_value not in ocr_result_dictionary:
            ocr_result_dictionary[key_value] = value
    # Extracting OCR last name, first name & Student ID
    last_name = ocr_result_dictionary['last name']
    first_name = ocr_result_dictionary['first name']
    student_id = ocr_result_dictionary['student number']
    ocr_student_identity_list = [last_name, first_name, student_id]
    #----
    # un-comment the following lines to see intermidiatory results
    #----
    #print(ocr_result_dictionary)
    #print(last_name)
    #print(first_name)
    #print(student_id)
    
    # Creating a dictionary of scores extracted from the image after OCR
    ocr_scors_dict = {}
    for key, value in ocr_result_dictionary.items():
         if key.startswith('Q') or key == 'total':
              ocr_scors_dict[key] = value
    #print(ocr_scors_dict)

    # Return Scors as a dictionary & Student details from OCR (first name, last name, student ID) as a list
    return ocr_scors_dict, ocr_student_identity_list

# From the returned list containing first name, last name and student id - extract first name only
def extract_ocr_firstname(ocr_student_identity):
     """this functions extracts the frist name from
     the ocr'ed list of student details"""
     first_name = ocr_student_identity[1]
     return first_name
# From the returned list containing first name, last name and student id - extract last name only
def extract_ocr_lastname(ocr_student_identity):
     """this functions extracts the last name from
     the ocr'ed list of student details"""
     last_name = ocr_student_identity[0]
     return last_name
# From the returned list containing first name, last name and student id - extract Student ID only
def extract_ocr_studentid(ocr_student_identity):
     """this functions extracts the student id from
     the ocr'ed list of student details """
     student_id = ocr_student_identity[2]
     return student_id
# Validate the scores - If the total mentiond in the image is correctly added or not
def calculation_validation(ocr_scores_dict):
    """this functions helps calculate totals or
    raise errors in case of calculation mistakes"""
    calulated_total = 0
    for key, value in ocr_scores_dict.items():
        ocr_extracted_total = 0
        if key.startswith('Q'):
            calulated_total += float(ocr_scores_dict[key])

    ocr_extracted_total = float(ocr_scores_dict['total'])

    if calulated_total != ocr_extracted_total:
        return False
    else:
        return calulated_total
# Fetched Student Details from a Central Student Data Excel Sheet
# USed for later comparison of OCR results
def fetch_student_details(CENTRAL_DATA_FILE):
    """this functions reads in the central Student Databse
    COnverts it into a dictionarty and returns the dictionary
    where the Student ID is the Key"""
    def create_student_dict(central_data_file):
        """this functions converts the student database 
        into a dictionary with the student id as the key"""
        df = pd.read_excel(central_data_file) # Reads the excel file into the program
        student_data_list = df.values.tolist() # Converts to a list
            
        student_dict = {}
        for sublist in student_data_list:
            key = sublist[3]
            value = [sublist[1], sublist[2]]
            student_dict[key] = value
        return student_dict
    main_student_dict = create_student_dict(CENTRAL_DATA_FILE)

    return main_student_dict
# The following functions perfroms logical operations to match OCR results with Central Student Database
def logical_operations(ocr_student_identity, university_data_dict):
    """This functions matches the closest True student ID
    and names from the central databse in the case of partical extractions"""

    # OCR results
    ocr_last_name = list(ocr_student_identity[1])
    ocr_first_name = list(ocr_student_identity[0])
    ocr_student_id = list(ocr_student_identity[2]) 
    
    missing_ocr_data = check_for_missing_ocrdata(ocr_last_name, ocr_first_name, ocr_student_id)
    # Error box in GUI in case OCR data is missing
    if missing_ocr_data == True:
        messagebox.showwarning("warning", "Missing Student Details, please click the next button")
    # This Section of the code should proceed only after the
    list_of_keys = creat_list_of_keys(university_data_dict)
    get_closest_match = check_closest_match(list_of_keys, ocr_student_id)
    return_index_of_match = get_index(get_closest_match)
    match_index_extract_as_list = pull_data(return_index_of_match, university_data_dict) # Use this to enter values into an excel sheet.
    compair_both_lists = compair_data(match_index_extract_as_list, ocr_student_identity)
    show_error_boolean = boolean_output( compair_both_lists)
    return show_error_boolean, match_index_extract_as_list
  
    

# These Functions and Nested Functions Return True if OCR extracted Data is Missing
# If OCR extracted data is not missing- They Return False
# False - No Data Missing - Proceed.
def check_for_missing_ocrdata(ocr_last_name, ocr_first_name, ocr_student_id):
    """this function checks if any of the ocr data is missing
    if it is missing, then it returns false else it returns true"""
    def check_lastname_empty(ocr_last_name):
        """this functions checks if the last name of of the form
        'E', 'M', 'P, 'T,'y'. If it is then it returns True
        else it returns False"""
        compair_list = ['E', 'M', 'P', 'T', 'Y']
        #print(ocr_last_name)
        if ocr_last_name == compair_list:
            return True
        return False
    
    def check_firstname_empty(ocr_first_name):
        """this functions checks if the first name of of the form
        'E', 'M', 'P, 'T,'y'. If it is then it returns True
        else it returns False"""
        compair_list = ['E', 'M', 'P', 'T', 'Y']
        #print(ocr_first_name)
        if ocr_first_name == compair_list:
            return True
        return False
    
    def check_studentid_empty(ocr_student_id):
        """this functions checks if the student id is of the form
        'E', 'M', 'P, 'T,'y'. If it is then it returns True
        else it returns False"""
        compair_list = ['E', 'M', 'P', 'T', 'Y']
        #print(ocr_student_id)
        if ocr_student_id ==  compair_list:
            return True
        return False
    # The following three variables hold boolean values returned from the previous functions
    missing_last_name = check_lastname_empty(ocr_last_name)
    missing_first_name = check_firstname_empty(ocr_first_name)
    missing_student_id = check_studentid_empty(ocr_student_id)
    # Specifies the condition of boolean output based on the boolean value returned from the previous functions
    if missing_student_id == True:
        return True
    elif missing_first_name == True or missing_last_name == True:
        return True
    return False
# Extracts all the key into a list from the master student data converted into a dictionary
# Keys only
# The Keys are all Student IDs
def creat_list_of_keys(university_data_dict):
    """creates a list of keys"""
    key_list = list(university_data_dict.keys())
    return key_list
# Checks for the closest matching student ID
def check_closest_match(list_of_keys, ocr_student_id):
    """check for closest match"""
    count_list = []
    for id in list_of_keys:
        individual_char_list = list(id)
        count = 0
        for char, letter in zip( individual_char_list, ocr_student_id):
            if char != letter:
                count += 1
        count_list.append(count)
    return count_list
# Returns the index value for of the key (student ID) that is the closest match
# This index value is used to extract key and value pairs from the master student databse
def get_index(count_list):
    """this function returns the index_value of the closest mathing ID"""
    min_index = 0
    min_value = count_list[0]
    for i in range(1, len(count_list)):
        if count_list[i] < min_value:
            min_value = count_list[i]
            min_index = i
    return min_index
# Based on the matching student ID and its Index
# The correct Student details are extracted from the Master Student Data Dictionary
def pull_data(return_index_of_match, university_data_dict):
    """this function looks for the key and value at index position
    and returns a list where the key and values are in the
    single list"""
    result_list = []
    correct_list = []
    key_value =  list(university_data_dict.keys())[return_index_of_match]
    result_list.append(key_value)
    value_list = university_data_dict[ key_value]
    for value in value_list:
        result_list.append(value)

    first_name = result_list[1]
    last_name = result_list[2]
    student_id = result_list[0]
    correct_list.append(first_name)
    correct_list.append(last_name)
    correct_list.append(student_id)
    return correct_list

# The following function compairs the student details
def compair_data(match_index_extract_as_list, ocr_student_identity):
    """this function compairs the ocr extracted first name, last name and student id
    with that in the univetsity student database"""
    missmatch_list = []
    for name, identity in zip(match_index_extract_as_list, ocr_student_identity):
        name = list(name)
        identity = list(identity)
        count = 0
        for char, letter in zip(name, identity):
            if char!=letter:
                count += 1
        missmatch_list.append(count)

    return missmatch_list
# This functions specifies the threshold limit for matches
# If the threshold is exceeded - it returns false - No match
# If threshold is within the limit - Match - proceed - True
def boolean_output(compair_both_lists):
    """this function returns a boolean output"""
    first_name = compair_both_lists[0]
    last_name = compair_both_lists[1]
    id = compair_both_lists[2]
    if first_name > 2 or last_name > 2:
        return False
    elif id > 1:
        return False
    return True
# This functions returns the scores & Total value to be entered into the Data Entry Sheet.
def prep_finaldata_for_dataentry(scores_dict):
    """test code before integration into the main code"""
    total = 0
    scors_entry_dict = {}
    for key, value in scores_dict.items():
        if key.startswith('Q'):
            scors_entry_dict[key] = value
    if 'total' in scors_entry_dict:
        total = float(scors_entry_dict['total'])
    return scors_entry_dict, total
            
# The following function Reads in the images folder
def load_image_filepath():
    """this function first loads the image filepath to be processed
    Creates a file dialoge box to enable the user to selct a folder
    from the computer"""
    # References the global variable
    global IMAGES_FOLDER_PATH, IMAGES_LIST, CURRENT_INDEX
    IMAGES_FOLDER_PATH = filedialog.askdirectory(title="Select Images Folder")
    if IMAGES_FOLDER_PATH:
        IMAGES_LIST = [os.path.join(IMAGES_FOLDER_PATH, f)
        for f in os.listdir(IMAGES_FOLDER_PATH)
        if f.lower().endswith(('.png', '.jpg', 'jpeg'))] # Specifies the types of image files to be read

        IMAGES_LIST.sort()
        CURRENT_INDEX = 0
        if IMAGES_LIST:
            display_image(IMAGES_LIST[0])
    
# This cycles to the next image when the next image button is pressed
def next_image():
    """this function is linked to the next image button that
    when clicked, opens up a new image"""
    # Referencing the global index
    global CURRENT_INDEX
    if IMAGES_LIST:
        CURRENT_INDEX = (CURRENT_INDEX + 1) % len(IMAGES_LIST)
        display_image(IMAGES_LIST[CURRENT_INDEX])

# The following function dislpays the image into the second frame of the GUI
def display_image(image_path):
    """this function helps display the image in frame 2"""
    entry_images_folder.delete(0, tk.END) # delete the frame before displaying the next image
    entry_images_folder.insert(0, image_path) # Insert the new image
    for widget in frame2.winfo_children():
        widget.destroy()
    # Get Frame 2 widht and hight for resizing later
    frame2_width = frame2.winfo_width # Returns the image display frame's Width
    frame2_height = frame2.winfo_height # Returns the image display frame's Height
    Scanned_image = document_scanner(image_path)
    wrap_image_pil, _ = resize(Scanned_image, width=500) # Resizes the image to fit the frame - Calls the resize function
    # Convert the image to a format that can be displayed in Tkinter
    wrap_image_pil = Image.fromarray(cv2.cvtColor(wrap_image_pil, cv2.COLOR_BGR2RGB))
    wrap_image_tk = ImageTk.PhotoImage(wrap_image_pil)
    image_label = ttk.Label(frame2, image=wrap_image_tk)
    image_label.image = wrap_image_tk  # Keep a reference to avoid garbage collection
    image_label.pack()
    ocr_scores_dict, ocr_student_identity = perform_ocr(Scanned_image) # Call to the OCR function
    ocr_first_name = extract_ocr_firstname(ocr_student_identity) # Call to the extract_ocr_firstname function
    ocr_last_name = extract_ocr_lastname(ocr_student_identity) # Call to the extract_ocr_lastname
    ocr_student_id = extract_ocr_studentid(ocr_student_identity) # Call to the extract_ocr_studentid function
    validate_scores = calculation_validation(ocr_scores_dict) # Call to the calculation_validation function
    if validate_scores == False:
        messagebox.showwarning("warning", "Missmatch in Totals, Please proceed to the next image") # Show warning box in case of missmatch of totals
    # call to the etch_student_details function
    university_data_dict = fetch_student_details(MASTER_DATA_GLOBAL_VARIABLE_FILEPATH)
    # Call to the logical_operations function
    validate_id_and_names, details_list = logical_operations(ocr_student_identity, university_data_dict)
    entry_first_name = details_list[0] # Extracts the firs name from the list returned
    entry_last_name = details_list[1]# Extracts the last name from the list returned
    entry_student_id = details_list[2] # # Extracts the Student ID from the list returned
    # Call to the data entry function
    final_scores_dict, _ = prep_finaldata_for_dataentry(ocr_scores_dict)
    first_name_entry.delete(0, tk.END) # Delete previous entry before displaying a new one
    first_name_entry.insert(0, entry_first_name) # Enter New value into the GUI display field
    last_name_entry.delete(0, tk.END) # Delete previous entry before displaying a new one
    last_name_entry.insert(0, entry_last_name) # Enter New value into the GUI display field
    student_id_entry.delete(0, tk.END) # Delete previous entry before displaying a new one
    student_id_entry.insert(0, entry_student_id) # Enter New value into the GUI display field
    total_marks_entry.delete(0, tk.END) # Delete previous entry before displaying a new one
    total_marks_entry.insert(0, validate_scores) # Enter New value into the GUI display field
    
    individual_scores_text.delete('1.0', tk.END)  # Clear existing content
    for key, value in final_scores_dict.items():
        individual_scores_text.insert(tk.END, f"{key}: {value}\n")

    return image_path, final_scores_dict, validate_scores

 
# Reading in the Excel file for entering the data into it.
def load_dataentry_excel_file():
    """this functions loads a blank or a partially filled excel sheet
    in which the extrated data needs to be filled"""
    global DATA_ENTRY_GLOBAL_VARIABLE_FILEPATH
    filepath_blank = filedialog.askopenfilename(title="Select an Excel Sheet",
                                                filetypes = (("Excel files", "*.xlsx"), ("All files", "*.*")))
    
    if filepath_blank:
        DATA_ENTRY_GLOBAL_VARIABLE_FILEPATH = filepath_blank
        entry_blank_excel.delete(0, tk.END)
        entry_blank_excel.insert(0, filepath_blank)
        return DATA_ENTRY_GLOBAL_VARIABLE_FILEPATH

# Reading in the Master Student Data file
def master_data_file():
    """this function loads the master data or the central university data
    excel sheet with serves as the central databse for comparison of
    extracted text later"""
    global MASTER_DATA_GLOBAL_VARIABLE_FILEPATH
    filepath_master = filedialog.askopenfilename(title="Select an Excel Sheet",
                                                filetypes = (("Excel files", "*.xlsx"), ("All files", "*.*")))
    if filepath_master:
        MASTER_DATA_GLOBAL_VARIABLE_FILEPATH = filepath_master
        entry_master_data.delete(0, tk.END)
        entry_master_data.insert(0, filepath_master)
        return  MASTER_DATA_GLOBAL_VARIABLE_FILEPATH
# Function to insert values into an excel sheet
def insert_values_to_excel():
    """this function loads the values to an excel sheet"""
    firstname = first_name_entry.get()
    lastname = last_name_entry.get()
    studentid = student_id_entry.get()
    total = total_marks_entry.get()

    path = DATA_ENTRY_GLOBAL_VARIABLE_FILEPATH
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    row_values = [firstname, lastname, studentid, total]
    sheet.append(row_values)
    workbook.save(path)
    # Insert rows into tree view


# Frames - GUI Main Frames specifying them, their dimensions & border widths
frame1 = ttk.Labelframe(window, text="Load Files", width=501, height=667, borderwidth=10)
frame11 = ttk.Frame(frame1)
frame12 = ttk.Frame(frame1)
frame13 = ttk.Frame(frame1)
frame2 = ttk.Labelframe(window, text="Image Display", width=501, height=667, borderwidth=10)
frame3 = ttk.Labelframe(window, text="Student Details and Scores", width=501, height=667, borderwidth=10)
frame31 = ttk.LabelFrame(frame3, text="Student Credentials", borderwidth=5)
frame32 = ttk.Labelframe(frame3, text="Individual Question Scores", borderwidth=5)
frame33 = ttk.Frame(frame3)

# Frame Positioning - Placing the GUI frames in an order by our preference
frame1.grid(row=0, column=0, padx=5, pady=5)
frame11.grid(row=1, column=0, padx=5, pady=50)
frame12.grid(row=2, column=0, padx=5, pady=100)
frame13.grid(row=3, column=0, padx=5, pady=50)
frame2.grid(row=0, column=1, padx=5, pady=5)
frame3.grid(row=0, column=2, padx=5, pady=5)
frame31.grid(row=0, column=0, padx=5, pady=10)
frame32.grid(row=1, column=0, padx=5, pady=52)
frame33.grid(row=2, column=0, padx=5, pady=10)

#Frame 1: Frame 11 elements
load_blank_excel_label = ttk.Label(frame11, text="Enter Data Entry File") # --> label
load_blank_excel_label.pack()
entry_blank_excel = ttk.Entry(frame11) # --> Entry Field
entry_blank_excel.pack()
load_blank_excel = ttk.Button(frame11, text="Load Data Entry File", command=load_dataentry_excel_file) # --> Button
load_blank_excel.pack()

#Frame 1: Frame 12 elements
load_master_data_label = ttk.Label(frame12, text="Enter University Master Data File") # --> label
load_master_data_label.pack()
entry_master_data = ttk.Entry(frame12) # --> Entry Field
entry_master_data.pack()
load_master_data_button = ttk.Button(frame12, text="Load University Data File", command=master_data_file) # --> Button
load_master_data_button.pack()

#Frame 1: Frame 13 elements
load_images_folder_label  = ttk.Label(frame13, text="Load Marksheet Images Folder")
load_images_folder_label.pack()
entry_images_folder = ttk.Entry(frame13)
entry_images_folder.pack()
load_images_folder_button = ttk.Button(frame13, text="Load Images Folder", command=load_image_filepath)
load_images_folder_button.pack()

# Frame 3: Frames and elements
first_name_label = ttk.Label(frame31, text="First Name") # --> label
first_name_label.grid(row=0, column=0, padx=2)
first_name_entry = ttk.Entry(frame31) # --> Entry Fiels
first_name_entry.grid(row=0, column=1, padx=2) 

last_name_label = ttk.Label(frame31, text="Last Name") # --> label
last_name_label.grid(row=1, column=0, padx=2)
last_name_entry = ttk.Entry(frame31) # --> Entry Fiels
last_name_entry.grid(row=1, column=1, padx=2) 

student_id_label = ttk.Label(frame31, text="Student ID") # --> label
student_id_label.grid(row=2, column=0, padx=2)
student_id_entry = ttk.Entry(frame31) # --> Entry Fiels
student_id_entry.grid(row=2, column=1, padx=2) 

student_id_label = ttk.Label(frame31, text="Student ID") # --> label
student_id_label.grid(row=2, column=0, padx=2)
student_id_entry = ttk.Entry(frame31) # --> Entry Fiels
student_id_entry.grid(row=2, column=1, padx=2) 

total_marks_label = ttk.Label(frame31, text= "Total Score") #--> Label
total_marks_label.grid(row=3, column=0, padx=2)
total_marks_entry = ttk.Entry(frame31) # --> Entry Fiels
total_marks_entry.grid(row=3, column=1, padx=2)

#Frame 3: Frame 32 elements
individual_scores_label = ttk.Label(frame32, text="Individual Question Scores")
individual_scores_label.grid(row=1, column=0, padx=2)
individual_scores_text = tk.Text(frame32, height=20, width= 30)
individual_scores_text.grid(row = 2, column=0, padx=2, pady=1)

#Frame 3: Frame 33 elements
next_image_button = ttk.Button(frame33, text="Next Image", command=next_image)
next_image_button.grid(row=0, column=0, padx=2, pady=5)

data_entry_button = ttk.Button(frame33, text="Enter Data To Excel", command=insert_values_to_excel)
data_entry_button.grid(row=0, column=1, padx=2, pady=10)
window.mainloop()