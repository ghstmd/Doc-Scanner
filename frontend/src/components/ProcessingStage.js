import { Box, LinearProgress, Typography } from '@mui/material';
import React from 'react';

const ProcessingStage = ({ progress, message }) => (
  <Box sx={{ width: '60%', mx: 'auto', mt: 10 }}>
    <Typography variant="h6" gutterBottom>{message}</Typography>
    <LinearProgress variant="determinate" value={progress} />
    <Typography mt={2}>{progress}%</Typography>
  </Box>
);

export default ProcessingStage;
