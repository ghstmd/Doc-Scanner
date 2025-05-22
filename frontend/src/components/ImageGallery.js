import { Box, Button, Typography } from '@mui/material';
import React from 'react';

const ImageGallery = ({ images, onRunOCR, onBackToUpload }) => {
  // Helper function to ensure URLs have the server base
  const getFullImageUrl = (url) => {
    if (url.startsWith('/')) {
      return `http://localhost:5000${url}`;
    }
    return url;
  };

  console.log("Images in gallery:", images);

  return (
    <Box>
      <Typography variant="h6">Review Processed Images:</Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
        {images.map((img, idx) => (
          <Box key={idx} sx={{ border: '1px solid #ddd', borderRadius: '4px', padding: '8px' }}>
            <img 
              src={getFullImageUrl(img.url)} 
              alt={img.name || `Image ${idx+1}`} 
              width={200} 
              style={{ display: 'block', marginBottom: '8px' }}
            />
            <Typography variant="caption">{img.name || `Image ${idx+1}`}</Typography>
          </Box>
        ))}
      </Box>
      <Box mt={3}>
        <Button variant="outlined" onClick={onBackToUpload}>Back</Button>
        <Button variant="contained" onClick={onRunOCR} sx={{ ml: 2 }}>Run OCR</Button>
      </Box>
    </Box>
  );
};

export default ImageGallery;