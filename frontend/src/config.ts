const config = {
    development: {
        API_BASE: 'http://localhost:8000',
        WS_BASE: 'ws://localhost:8000'
    },
    production: {
        API_BASE: 'https://your-backend-url.railway.app', // you'll update this after backend deployment
        WS_BASE: 'wss://your-backend-url.railway.app'    // you'll update this after backend deployment
    }
};

const environment = process.env.NODE_ENV || 'development';
export default config[environment as keyof typeof config];