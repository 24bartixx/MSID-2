from pathlib import Path

DATA_PATH = Path("..") / "data" / "data.csv"
 
CATEGORICAL_COLUMN_NAMES = [
     "Marital status",
     "Application mode",
     "Application order",
     "Course",
     "Daytime/evening attendance",
     "Previous qualification",
     "Nacionality",
     "Mother's qualification",
     "Father's qualification",
     "Mother's occupation",
     "Father's occupation",
     "Displaced",
     "Educational special needs",
     "Debtor",
     "Tuition fees up to date",
     "Gender",
     "Scholarship holder",
     "International",
     "Target"
]

SEM_1ST_COLUMN_NAMES = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)"
]
 
MATRIAL_STATUS_TRANSLATION = {
    1: "single",
    2: "married",
    3: "widower",
    4: "divorced",
    5: "facto union",
    6: "legally separated"
}

APPLICATION_MODE_MAP = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}
 
APPLICATION_ORDER_MAP = {
    0: "1st choice",
    1: "2nd choice",
    2: "3rd choice",
    3: "4th choice",
    4: "5th choice",
    5: "6th choice",
    6: "7th choice",
    7: "8th choice",
    8: "9th choice",
    9: "10th choice"
}

COURSE_MAP = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

ATTENDANCE_MAP = {
    1: "Daytime",
    0: "Evening"
}

PREVIOUS_QUALIFICATION_MAP = {
    1: "Secondary education",
    2: "Bachelor's degree",
    3: "Higher ed - degree",
    4: "Higher ed - master's",
    5: "Higher ed - doctorate",
    6: "Higher ed (incomplete)",
    9: "12th year - not completed",
    10: "11th year - not completed",
    12: "Other",
    14: "10th year",
    15: "10th year - not completed",
    19: "Basic ed 3rd cycle (9th/10th/11th)",
    38: "Basic ed 2nd cycle (6th–8th)",
    39: "Basic ed 2nd cycle (6th–8th) equiv.",
    40: "Tech specialization course",
    41: "Higher ed - degree (1st cycle)",
    42: "Professional tech course",
    43: "Higher ed - master (2nd cycle)"
}

NACIONALITY_MAP = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santoméan",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldovan",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

MOTHER_QUALIFICATION_MAP = {
    1: "Secondary Education - 12th Year or Eq.",
    2: "Higher Ed - Bachelor's",
    3: "Higher Ed - Degree",
    4: "Higher Ed - Master's",
    5: "Higher Ed - Doctorate",
    6: "Frequency of Higher Ed",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year",
    14: "10th Year",
    18: "General Commerce",
    19: "Basic Ed 3rd Cycle",
    22: "Technical-Professional",
    26: "7th Year",
    27: "2nd Cycle of General High School",
    29: "9th Year - Not Completed",
    30: "8th Year",
    34: "Unknown",
    35: "Can't Read or Write",
    36: "Can Read (no 4th year)",
    37: "Basic Ed 1st Cycle (4th/5th)",
    38: "Basic Ed 2nd Cycle (6th–8th)",
    39: "Tech Specialization",
    40: "Higher Ed - Degree (1st Cycle)",
    41: "Specialized Higher Studies",
    42: "Professional Higher Tech Course",
    43: "Higher Ed - Master's (2nd Cycle)",
    44: "Higher Ed - Doctorate (3rd Cycle)"
}

FATHER_QUALIFICATION_MAP = {
    1: "Secondary Ed - 12th Year or Eq.",
    2: "Higher Ed - Bachelor's",
    3: "Higher Ed - Degree",
    4: "Higher Ed - Master's",
    5: "Higher Ed - Doctorate",
    6: "Frequency of Higher Ed",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year",
    13: "2nd Comp. High School Course",
    14: "10th Year",
    18: "General Commerce Course",
    19: "Basic Ed 3rd Cycle",
    20: "Complementary High School",
    22: "Technical-Professional Course",
    25: "Comp. High School - Not Concluded",
    26: "7th Year",
    27: "2nd Cycle of General HS",
    29: "9th Year - Not Completed",
    30: "8th Year",
    31: "General Admin & Commerce",
    33: "Supplementary Admin & Accounting",
    34: "Unknown",
    35: "Can't Read or Write",
    36: "Can Read (no 4th Year)",
    37: "Basic Ed 1st Cycle (4th/5th)",
    38: "Basic Ed 2nd Cycle (6th–8th)",
    39: "Technological Spec. Course",
    40: "Higher Ed - Degree (1st Cycle)",
    41: "Specialized Higher Studies",
    42: "Professional Higher Tech",
    43: "Higher Ed - Master (2nd Cycle)",
    44: "Higher Ed - Doctorate (3rd Cycle)"
}

MOTHER_OCCUPATION_MAP = {
    0: "Student",
    1: "Legislative Power & Executives",
    2: "Scientific Specialists",
    3: "Technicians and Professions",
    4: "Administrative Staff",
    5: "Personal Services & Security Workers",
    6: "Farmers and Skilled Agricultural Workers",
    7: "Skilled Industry Workers",
    8: "Machine Operators and Assemblers",
    9: "Unskilled Workers",
    10: "Armed Forces",
    90: "Other Situation",
    99: "(Blank)",
    122: "Health Professionals",
    123: "Teachers",
    125: "ICT Specialists",
    131: "Engineering Technicians (Intermediate)",
    132: "Health Technicians (Intermediate)",
    134: "Technicians - Social, Sports, Cultural",
    141: "Office & Data Processing Workers",
    143: "Accounting & Registry Operators",
    144: "Other Admin Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers",
    171: "Skilled Construction Workers",
    173: "Printing & Precision Craft Workers",
    175: "Woodworking, Textile, Crafts Workers",
    191: "Cleaning Workers",
    192: "Unskilled Agricultural Workers",
    193: "Unskilled Industry/Construction Workers",
    194: "Meal Preparation Assistants"
}

FATHER_OCCUPATION_MAP = {
    0: "Student",
    1: "Legislative Power & Executives",
    2: "Scientific Specialists",
    3: "Technicians and Professions",
    4: "Administrative Staff",
    5: "Personal Services & Security Workers",
    6: "Farmers and Skilled Agricultural Workers",
    7: "Skilled Industry Workers",
    8: "Machine Operators and Assemblers",
    9: "Unskilled Workers",
    10: "Armed Forces",
    90: "Other Situation",
    99: "(Blank)",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces Personnel",
    112: "Directors of Admin and Commercial Services",
    114: "Hotel, Catering, Trade Directors",
    121: "Physical Sciences & Engineering Specialists",
    122: "Health Professionals",
    123: "Teachers",
    124: "Finance & Admin Specialists",
    131: "Intermediate Level Engineering Technicians",
    132: "Health Technicians",
    134: "Technicians - Social, Sports, Culture",
    135: "ICT Technicians",
    141: "Office Workers & Data Operators",
    143: "Accounting & Registry Operators",
    144: "Other Admin Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers",
    154: "Security Services Personnel",
    161: "Market-Oriented Farmers",
    163: "Livestock/Fishery/Subsistence Workers",
    171: "Skilled Construction Workers",
    172: "Metal & Metallurgy Workers",
    174: "Electric/Electronic Technicians",
    175: "Wood/Furniture/Textile Workers",
    181: "Fixed Plant and Machine Operators",
    182: "Assembly Workers",
    183: "Vehicle Drivers and Mobile Operators",
    192: "Unskilled Agricultural Workers",
    193: "Unskilled Industry/Construction Workers",
    194: "Meal Preparation Assistants",
    195: "Street Vendors (except food)"
}

BINARY_MAP = {
    0: "No",
    1: "Yes"
}

GENDER_MAP = {
    0: "Female",
    1: "Male"
}

CATEGORY_TRANSLATIONS = {
    "Marital status": MATRIAL_STATUS_TRANSLATION,
    "Application mode": APPLICATION_MODE_MAP,
    "Application order": APPLICATION_ORDER_MAP,
    "Course": COURSE_MAP,
    "Daytime/evening attendance": ATTENDANCE_MAP,
    "Previous qualification": PREVIOUS_QUALIFICATION_MAP,
    "Nacionality": NACIONALITY_MAP,
    "Mother's qualification": MOTHER_QUALIFICATION_MAP,
    "Father's qualification": FATHER_QUALIFICATION_MAP,
    "Mother's occupation": MOTHER_OCCUPATION_MAP,
    "Father's occupation": FATHER_OCCUPATION_MAP,
    "Displaced": BINARY_MAP,
    "Educational special needs": BINARY_MAP,
    "Debtor": BINARY_MAP,
    "Tuition fees up to date": BINARY_MAP,
    "Gender": GENDER_MAP,
    "Scholarship holder": BINARY_MAP,
    "International": BINARY_MAP
}
 