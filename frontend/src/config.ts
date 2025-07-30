const config = {
    development: {
        API_BASE: 'http://localhost:8000',
        WS_BASE: 'ws://localhost:8000'
    },
    production: {
        API_BASE: 'https://therapist-backend.onrender.com', // replace with actual render URL
        WS_BASE: 'wss://therapist-backend.onrender.com'     // replace with actual render URL
    }
};

const environment = process.env.NODE_ENV || 'development';
export default config[environment as keyof typeof config];
