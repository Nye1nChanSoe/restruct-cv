"""Semantic reference text used by MiniLM classifiers and debug colors."""

SECTION_REFERENCES: dict[str, tuple[str, ...]] = {
    "summary": (
        "Professional Summary", "Career Profile", "Career Objective", "About Me",
        "Summary", "Profile", "Executive Summary", "Personal Statement", "Objective",
        "Professional Profile", "Career Summary", "Overview", "Highlights",
        "Key Qualifications", "Profile Summary", "Profile"
    ),
    "experience": (
        "Work Experience", "Professional Experience", "Employment History",
        "Career History", "Experience", "Relevant Experience", "Work History",
        "Job History", "Employment", "Positions Held", "Field Experience",
        "Industry Experience", "Design Experience", "Marketing Experience",
        "Freelance Experience", "Client Experience", "Internship Experience",
        "Practical Experience",
    ),
    "education": (
        "Education", "Academic Background", "Academic Qualifications",
        "Education and Training", "Educational Background", "Academic History",
        "Schooling", "Degrees", "Coursework", "Relevant Coursework",
    ),
    "skills": (
        "Skills", "Technical Skills", "Core Competencies", "Technologies and Tools",
        "Key Skills", "Skill Set", "Areas of Expertise", "Competencies",
        "Software Skills", "Design Skills", "Design Tools", "Programming Languages",
        "Languages and Frameworks", "Tech Stack", "Tools and Technologies",
        "Marketing Skills", "Digital Skills", "Hard Skills", "Soft Skills",
        "Proficiencies", "Specializations", "UX Skills", "UI Skills", "Skills Summary",
    ),
    "projects": (
        "Projects", "Selected Projects", "Personal Projects", "Portfolio",
        "Case Studies", "Design Portfolio", "Featured Projects", "Key Projects",
        "Project Highlights", "Campaigns", "Selected Campaigns", "Notable Work",
        "Side Projects", "Open Source Contributions", "GitHub Projects",
        "Academic Projects",
    ),
    "certifications": (
        "Certifications", "Certificates and Training", "Professional Certifications",
        "Certificates", "Licenses and Certifications", "Training and Certifications",
        "Professional Development", "Credentials", "Accreditations",
    ),
    "licenses": (
        "Licenses", "Driver's License", "Commercial License", "Trade License",
        "Permits and Licenses", "Professional Licenses", "CDL", "OSHA Certification",
        "Safety Certifications", "Forklift Certification",
    ),
    "tools_equipment": (
        "Tools and Equipment", "Equipment Operated", "Machinery",
        "Equipment Proficiency", "Tools", "Equipment Experience", "Machine Operation",
    ),
    "languages": (
        "Languages", "Language Proficiency", "Language Skills", "Spoken Languages",
    ),
    "volunteering": (
        "Volunteer Experience", "Community Involvement", "Community Service",
        "Volunteering", "Volunteer Work", "Civic Engagement",
    ),
    "awards": (
        "Awards and Honors", "Achievements", "Awards", "Honors", "Recognition",
        "Accolades", "Accomplishments",
    ),
    "publications": (
        "Publications", "Research and Publications", "Papers", "Articles Published",
        "Conference Talks", "Speaking Engagements", "Presentations",
    ),
    "references": (
        "References", "Professional References", "References Available Upon Request",
    ),
    "interests": (
        "Interests", "Hobbies", "Personal Interests", "Activities",
        "Extracurricular Activities",
    ),
}


SECTION_COLORS: dict[str, str] = {
    "summary": "#2E7D32",
    "experience": "#1565C0",
    "education": "#7B1FA2",
    "skills": "#00897B",
    "projects": "#EF6C00",
    "certifications": "#C2185B",
    "licenses": "#AD1457",
    "tools_equipment": "#00695C",
    "languages": "#558B2F",
    "volunteering": "#00838F",
    "awards": "#F9A825",
    "publications": "#6D4C41",
    "references": "#455A64",
    "interests": "#8D6E63",
    "others": "#607D8B",
}


# These are representative semantic prototypes, not canonical output values.
# Extracted titles retain their exact source text.
JOB_TITLE_REFERENCES: dict[str, tuple[str, ...]] = {
    "software_engineering": (
        "Software Engineer", "Software Developer", "Application Developer",
        "Backend Engineer", "Backend Developer", "Frontend Engineer",
        "Frontend Developer", "Full Stack Engineer", "Full Stack Developer",
        "Mobile Developer", "iOS Developer", "Android Developer", "Game Developer",
        "Embedded Software Engineer", "Systems Programmer", "Web Developer",
        "API Developer", "Solutions Engineer",
    ),
    "cloud_devops_security": (
        "Cloud Engineer", "Cloud Architect", "DevOps Engineer", "DevSecOps Engineer",
        "Site Reliability Engineer", "Platform Engineer", "Infrastructure Engineer",
        "Systems Administrator", "Network Engineer", "Security Engineer",
        "Cybersecurity Analyst", "Information Security Specialist",
        "Penetration Tester", "SOC Analyst", "Database Administrator",
    ),
    "data_ai_quality": (
        "Data Analyst", "Data Engineer", "Data Scientist", "Machine Learning Engineer",
        "AI Engineer", "Research Scientist", "Business Intelligence Analyst",
        "Analytics Engineer", "Database Engineer", "Statistician", "QA Engineer",
        "Quality Assurance Analyst", "Software Test Engineer", "Automation Test Engineer",
    ),
    "product_project_management": (
        "Product Manager", "Product Owner", "Technical Product Manager",
        "Project Manager", "Program Manager", "Technical Program Manager",
        "Scrum Master", "Delivery Manager", "Implementation Manager",
        "Business Analyst", "Systems Analyst", "Management Consultant",
    ),
    "design_creative": (
        "UX Designer", "UI Designer", "UX UI Designer", "Product Designer",
        "Interaction Designer", "User Experience Researcher", "Visual Designer",
        "Graphic Designer", "Web Designer", "Motion Designer", "Industrial Designer",
        "Interior Designer", "Fashion Designer", "Art Director", "Creative Director",
        "Video Editor", "Photographer", "Illustrator", "Copywriter",
    ),
    "marketing_communications": (
        "Digital Marketing Specialist", "Digital Marketing Manager", "Marketing Manager",
        "Marketing Coordinator", "Growth Marketing Manager", "Performance Marketer",
        "Content Marketing Manager", "Content Strategist", "SEO Specialist",
        "SEM Specialist", "Social Media Manager", "Brand Manager",
        "Communications Manager", "Public Relations Specialist", "Media Planner",
        "Email Marketing Specialist", "Market Research Analyst",
    ),
    "sales_customer_success": (
        "Sales Representative", "Sales Manager", "Account Executive", "Account Manager",
        "Business Development Manager", "Business Development Representative",
        "Sales Engineer", "Solutions Consultant", "Customer Success Manager",
        "Customer Support Specialist", "Call Center Representative", "Retail Associate",
        "Real Estate Agent", "Insurance Agent",
    ),
    "finance_accounting": (
        "Accountant", "Senior Accountant", "Auditor", "Financial Analyst",
        "Finance Manager", "Controller", "Bookkeeper", "Tax Consultant",
        "Investment Analyst", "Portfolio Manager", "Risk Analyst", "Credit Analyst",
        "Bank Teller", "Payroll Specialist", "Actuary", "Economist",
    ),
    "operations_supply_chain": (
        "Operations Manager", "Operations Analyst", "Supply Chain Analyst",
        "Supply Chain Manager", "Logistics Coordinator", "Logistics Manager",
        "Procurement Specialist", "Purchasing Manager", "Inventory Planner",
        "Warehouse Manager", "Fleet Manager", "Demand Planner", "Production Planner",
        "Import Export Coordinator", "Quality Control Inspector",
    ),
    "administration_hr_legal": (
        "Administrative Assistant", "Executive Assistant", "Office Manager",
        "Receptionist", "Data Entry Clerk", "Human Resources Manager",
        "Human Resources Specialist", "Recruiter", "Talent Acquisition Specialist",
        "Training Coordinator", "Compensation Analyst", "Lawyer", "Attorney",
        "Legal Counsel", "Paralegal", "Legal Assistant", "Compliance Officer",
    ),
    "healthcare_clinical": (
        "Doctor", "Physician", "Surgeon", "General Practitioner", "Registered Nurse",
        "Nurse Practitioner", "Licensed Practical Nurse", "Medical Assistant",
        "Pharmacist", "Dentist", "Dental Assistant", "Physical Therapist",
        "Occupational Therapist", "Radiologic Technologist", "Medical Technologist",
        "Paramedic", "Emergency Medical Technician", "Psychologist", "Counselor",
        "Veterinarian", "Nutritionist", "Caregiver",
    ),
    "education_research": (
        "Teacher", "Primary School Teacher", "High School Teacher", "Lecturer",
        "Professor", "Teaching Assistant", "Tutor", "School Counselor",
        "Principal", "Academic Advisor", "Instructional Designer", "Librarian",
        "Research Assistant", "Researcher", "Laboratory Technician",
    ),
    "civil_mechanical_electrical": (
        "Civil Engineer", "Structural Engineer", "Geotechnical Engineer",
        "Mechanical Engineer", "Electrical Engineer", "Electronics Engineer",
        "Chemical Engineer", "Industrial Engineer", "Manufacturing Engineer",
        "Process Engineer", "Quality Engineer", "Automotive Engineer",
        "Aerospace Engineer", "Marine Engineer", "Environmental Engineer",
        "Biomedical Engineer", "Site Engineer", "Field Engineer", "Design Engineer",
        "Engineering Manager", "CAD Engineer", "Draftsperson",
    ),
    "construction_trades": (
        "Construction Manager", "Construction Worker", "Site Supervisor",
        "Quantity Surveyor", "Surveyor", "Architect", "Electrician", "Plumber",
        "Carpenter", "Welder", "Metal Fabricator", "Mason", "Painter",
        "Roofer", "HVAC Technician", "Refrigeration Technician", "Pipefitter",
        "Heavy Equipment Operator", "Crane Operator", "Safety Officer",
    ),
    "maintenance_manufacturing": (
        "Maintenance Technician", "Industrial Maintenance Technician",
        "Mechanical Technician", "Electrical Technician", "Service Technician",
        "Field Service Technician", "Machine Operator", "CNC Operator", "Machinist",
        "Production Operator", "Assembly Technician", "Plant Operator",
        "Instrumentation Technician", "Maintenance Mechanic", "Factory Worker",
    ),
    "transportation_logistics": (
        "Truck Driver", "Delivery Driver", "Bus Driver", "Taxi Driver", "Courier",
        "Forklift Operator", "Warehouse Associate", "Material Handler", "Dispatcher",
        "Freight Coordinator", "Pilot", "Flight Attendant", "Aircraft Mechanic",
        "Ship Captain", "Deck Officer", "Seafarer",
    ),
    "hospitality_food_service": (
        "Hotel Manager", "Front Desk Agent", "Concierge", "Housekeeper", "Chef",
        "Sous Chef", "Cook", "Baker", "Bartender", "Barista", "Server",
        "Restaurant Manager", "Food Service Worker", "Event Coordinator",
        "Travel Agent", "Tour Guide",
    ),
    "agriculture_environment": (
        "Farmer", "Farm Worker", "Agricultural Engineer", "Agronomist",
        "Horticulturist", "Landscaper", "Environmental Scientist", "Geologist",
        "Forester", "Fisheries Officer", "Water Resources Engineer",
    ),
    "public_service_safety": (
        "Police Officer", "Security Guard", "Firefighter", "Military Officer",
        "Social Worker", "Community Outreach Coordinator", "Government Officer",
        "Policy Analyst", "Urban Planner", "Postal Worker",
    ),
    "leadership_general": (
        "Chief Executive Officer", "Chief Technology Officer", "Chief Financial Officer",
        "Managing Director", "General Manager", "Department Manager", "Team Lead",
        "Supervisor", "Coordinator", "Specialist", "Consultant", "Associate",
        "Intern", "Trainee",
    ),
}


JOB_TITLE_NEGATIVE_REFERENCES: dict[str, tuple[str, ...]] = {
    "header_labels": (
        "Contact", "Contact Information", "Personal Details", "Personal Information",
        "Profile", "About Me", "Curriculum Vitae", "Resume", "Career Summary",
    ),
    "availability_and_status": (
        "Available for work", "Available for shift work", "Available for relocation",
        "Willing to relocate", "Open to opportunities", "Immediately available",
        "Currently seeking employment", "Full time availability", "Part time availability",
    ),
    "citizenship_and_work_rights": (
        "Thai citizen", "US citizen", "Permanent resident", "Work permit holder",
        "Authorized to work", "No sponsorship required", "Nationality Thai",
    ),
    "contact_and_location": (
        "Bangkok Thailand", "Singapore", "Yangon Myanmar", "Email address",
        "Phone number", "LinkedIn profile", "GitHub profile", "Personal website",
    ),
    "section_and_resume_text": (
        "Professional experience", "Education", "Technical skills", "References",
        "Languages", "Certifications", "Selected projects", "Employment history",
        "References available upon request",
    ),
    "descriptive_statements": (
        "Results driven professional", "Experienced team player",
        "Strong communication skills", "Hard working and reliable",
        "Able to work under pressure", "Seeking a challenging position",
    ),
}
