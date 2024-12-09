from fpdf import FPDF
import os
import re

class CodePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.current_filename = ''
        
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, self.current_filename, 0, 1, 'C')
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_code_pdf(files_dict, output_filename='code_documentation.pdf'):
    pdf = CodePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    for filename, content in files_dict.items():
        pdf.current_filename = filename
        pdf.add_page()
        
        # Add filename as header
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, filename, 0, 1, 'L')
        pdf.ln(5)
        
        # Add code content
        pdf.set_font('Courier', '', 9)
        
        # Remove line numbers and process content
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove line numbers if they exist
            match = re.match(r'^\d+\|(.*)$', line)
            if match:
                cleaned_lines.append(match.group(1))
            else:
                cleaned_lines.append(line)
        
        # Join lines and add to PDF
        code_text = '\n'.join(cleaned_lines)
        pdf.multi_cell(0, 5, code_text)
        pdf.ln(10)
    
    pdf.output(output_filename)

files_to_convert = {}
file_paths = ['inference.py', 'decoder_1.py', 'text_image_token_processor_1.py', 'vision_transformer_1.py']

for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            files_to_convert[file_path] = f.read()
    else:
        print(f"Warning: File {file_path} not found")

create_code_pdf(files_to_convert)