const config = {
    development: {
        API_BASE: 'http://localhost:8000',
        WS_BASE: 'ws://localhost:8000'
    },
    production: {
        API_BASE: 'https://therapist-backend-223632250356.us-east1.run.app', 
        WS_BASE: 'https://therapist-backend-223632250356.us-east1.run.app'    
    }
};

const environment = process.env.NODE_ENV || 'development';
export default config[environment as keyof typeof config];