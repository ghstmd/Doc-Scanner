import axios from "axios";
import React, { useState } from "react";
import FileUpload from "./FileUpload";
import ImageGallery from "./ImageGallery";
import OCRResults from "./OCRResults";
import ProcessingStage from "./ProcessingStage";

export default function Home() {
  const [files, setFiles] = useState([]);
  const [processedImages, setProcessedImages] = useState([]);
  const [ocrResult, setOcrResult] = useState(null);
  const [sessionId, setSessionId] = useState("");
  const [loading, setLoading] = useState(false);
  const [stageMessage, setStageMessage] = useState("");
  const [error, setError] = useState("");

  const handleFilesAdded = (newFiles) => {
    setFiles((prev) => [...prev, ...newFiles]);
  };

  const handleDeleteFile = (index) => {
    setFiles((prev) => prev.filter((_, idx) => idx !== index));
  };

  const handleMoveUp = (index) => {
    if (index === 0) return;
    const newFiles = [...files];
    [newFiles[index - 1], newFiles[index]] = [newFiles[index], newFiles[index - 1]];
    setFiles(newFiles);
  };

  const handleMoveDown = (index) => {
    if (index === files.length - 1) return;
    const newFiles = [...files];
    [newFiles[index + 1], newFiles[index]] = [newFiles[index], newFiles[index + 1]];
    setFiles(newFiles);
  };

  const handlePreprocess = async () => {
    if (!files.length) {
      return alert("Please select images first!");
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("images", file);
    });

    try {
      setLoading(true);
      setStageMessage("Preprocessing (Scanning) images...");
      setError("");

      const res = await axios.post("http://localhost:5000/api/preprocess", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const sid = res.data.session_id || res.data.sessionId;
      const images = res.data.images || res.data.processedImages;

      if (!sid) throw new Error("No session ID in response");
      if (!images || !Array.isArray(images)) throw new Error("Invalid images format");

      setSessionId(sid);
      setProcessedImages(images);
      setOcrResult(null);
    } catch (err) {
      console.error(err);
      if (err.response) {
        setError(`Server error: ${JSON.stringify(err.response.data)}`);
      } else if (err.request) {
        setError("No response from server. Is backend running?");
      } else {
        setError(`Error: ${err.message}`);
      }
      alert("Preprocessing failed: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleRunOCR = async () => {
    if (!sessionId) {
      return alert("No session found. Preprocess first!");
    }

    try {
      setLoading(true);
      setStageMessage("Performing OCR...");

      const res = await axios.post("http://localhost:5000/api/ocr", 
        { session_id: sessionId },
        { headers: { "Content-Type": "application/json" } }
      );

      if (!Array.isArray(res.data)) {
        throw new Error("Invalid OCR response, expected a list");
      }

      setOcrResult(res.data);
    } catch (err) {
      console.error(err);
      alert("OCR failed: " + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (imageId, format) => {
    if (!imageId) {
      return alert("Invalid image ID!");
    }
    window.open(`http://localhost:5000/api/download?id=${imageId}&format=${format}`, "_blank");
  };

  if (loading) {
    return (
      <div>
        <ProcessingStage progress={70} message={stageMessage} />
        {error && <div style={{ color: 'red', margin: '20px' }}>{error}</div>}
      </div>
    );
  }

  if (ocrResult) {
    return <OCRResults results={ocrResult} onDownload={handleDownload} />;
  }

  if (processedImages && processedImages.length > 0) {
    return (
      <ImageGallery
        images={processedImages}
        onRunOCR={handleRunOCR}
        onBackToUpload={() => {
          setFiles([]);
          setProcessedImages([]);
          setSessionId("");
        }}
      />
    );
  }

  return (
    <div style={{ padding: 20 }}>
      <h1>Document Scanner + OCR</h1>
      {error && <div style={{ color: 'red', margin: '20px' }}>{error}</div>}
      <FileUpload
        files={files}
        onFilesAdded={handleFilesAdded}
        onDeleteFile={handleDeleteFile}
        onMoveUp={handleMoveUp}
        onMoveDown={handleMoveDown}
        onProcessImages={handlePreprocess}
      />
    </div>
  );
}
