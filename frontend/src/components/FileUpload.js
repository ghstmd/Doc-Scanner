import { Box, Button } from '@mui/material';
import React from 'react';

const FileUpload = ({ files, onFilesAdded, onDeleteFile, onProcessImages, onMoveUp, onMoveDown }) => {
  const handleFileChange = (e) => {
    onFilesAdded(Array.from(e.target.files));
  };

  return (
    <Box>
      <input type="file" multiple onChange={handleFileChange} />
      <ul>
        {files.map((file, index) => (
          <li key={index}>
            {file.name}
            <Button onClick={() => onMoveUp(index)}>⬆️</Button>
            <Button onClick={() => onMoveDown(index)}>⬇️</Button>
            <Button onClick={() => onDeleteFile(index)}>❌</Button>
          </li>
        ))}
      </ul>
      <Button variant="contained" onClick={onProcessImages} disabled={files.length === 0}>Process Images</Button>
    </Box>
  );
};

export default FileUpload;
