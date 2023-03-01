# Khởi tạo PyPDF2
import PyPDF2

# Đọc tệp PDF
pdfFileObj = open('input.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# Lấy văn bản từ tệp PDF
text = ""
for pageNum in range(pdfReader.numPages):
    pageObj = pdfReader.getPage(pageNum)
    text += pageObj.extractText()

# Xử lý văn bản
text = text.replace("\n", " ")

# Dịch văn bản sang tiếng Việt
from googletrans import Translator
translator = Translator()
text_vi = translator.translate(text, dest='vi').text

# Ghi văn bản sang tệp PDF out.pdf
pdfWriter = PyPDF2.PdfFileWriter()

