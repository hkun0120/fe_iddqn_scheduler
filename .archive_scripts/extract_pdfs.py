#!/usr/bin/env python3
"""Extract title, abstract, and key sections from all PDFs in 1/ folder."""
import fitz  # PyMuPDF
import os, re, json

pdf_dir = "1/"
output = {}

for fname in sorted(os.listdir(pdf_dir)):
    if not fname.endswith(".pdf"):
        continue
    fpath = os.path.join(pdf_dir, fname)
    try:
        doc = fitz.open(fpath)
        num_pages = doc.page_count
        
        # Extract first 5 pages (title, abstract, intro, methodology overview)
        text_pages = []
        for i in range(min(5, num_pages)):
            text_pages.append(doc[i].get_text())
        
        # Also extract conclusion (usually last 2 pages)
        conclusion_text = []
        for i in range(max(0, num_pages - 2), num_pages):
            if i >= 5:  # avoid duplicates
                conclusion_text.append(doc[i].get_text())
        
        full_early = "\n".join(text_pages)
        full_late = "\n".join(conclusion_text)
        
        output[fname] = {
            "pages": num_pages,
            "early_text": full_early[:8000],
            "late_text": full_late[:4000],
        }
        doc.close()
        print(f"OK: {fname} ({num_pages} pages)")
    except Exception as e:
        import traceback
        print(f"ERR: {fname}: {e}")
        traceback.print_exc()
        output[fname] = {"error": str(e)}

# Save to JSON
with open("pdf_extracts.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nDone: {len(output)} PDFs processed. Saved to pdf_extracts.json")
