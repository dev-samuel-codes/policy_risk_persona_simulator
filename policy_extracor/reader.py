import os
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader
from docx import Document


# ==========================
# PDF 읽기
# ==========================

def read_pdf(path):

    reader = PdfReader(path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================
# DOCX 읽기
# ==========================

def read_docx(path):

    doc = Document(path)

    text = ""

    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    return text


# ==========================
# HWPX 읽기
# ==========================

def read_hwpx(path):

    text = ""

    with zipfile.ZipFile(path, "r") as z:

        for file in z.namelist():

            if file.endswith(".xml"):

                xml = z.read(file)

                root = ET.fromstring(xml)

                for node in root.iter():

                    if node.text:
                        text += node.text.strip() + "\n"

    return text


# ==========================
# HWP 읽기
# ==========================

def read_hwp(path):

    try:

        from hwp5.filestructure import Hwp5File

        hwp = Hwp5File(path)

        return hwp.text

    except Exception as e:

        raise Exception(
            f"HWP 파일을 읽을 수 없습니다.\n{e}"
        )


# ==========================
# 자동 파일 읽기
# ==========================

def read_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError("파일을 찾을 수 없습니다.")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return read_pdf(path)

    elif ext == ".docx":
        return read_docx(path)

    elif ext == ".hwpx":
        return read_hwpx(path)

    elif ext == ".hwp":
        return read_hwp(path)

    else:
        raise Exception("지원하지 않는 파일 형식입니다.")


if __name__ == "__main__":

    file_path = input("파일 경로 : ")

    text = read_file(file_path)

    print("=" * 60)
    print(text[:3000])
    print("=" * 60)