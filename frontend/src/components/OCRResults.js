import React from "react";

export default function OCRResults({ results, onDownload }) {
  return (
    <div style={{ padding: 20 }}>
      <h1>OCR Results</h1>
      {results.map((item, index) => (
        <div key={item.imageId || index} style={{ marginBottom: 30, border: "1px solid #ccc", padding: 15, borderRadius: 10 }}>
          <h3>{item.originalName || `Image ${index + 1}`}</h3>
          <textarea
            value={item.textPreview || ""}
            rows={10}
            style={{ width: "100%", marginBottom: 10 }}
            readOnly
          />
          <div>
            <button onClick={() => onDownload(item.imageId, "txt")} style={{ marginRight: 10 }}>Download TXT</button>
            <button onClick={() => onDownload(item.imageId, "docx")}>Download DOCX</button>
          </div>
        </div>
      ))}
    </div>
  );
}
