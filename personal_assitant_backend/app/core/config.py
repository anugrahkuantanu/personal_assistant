from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Personal Assistant API"
    DEBUG: bool = True
    CORS_ORIGINS: list = ["http://localhost:3000"]
    
    # API Keys
    OPENAI_API_KEY: str
    
    
    # Email server settings
    IMAP_SERVER: str = "imap.one.com"  # One.com IMAP server
    IMAP_PORT: int = 993
    SMTP_SERVER: str = "send.one.com"  # One.com SMTP server
    SMTP_PORT: int = 465
    
    # Calendar settings
    CALDAV_URL: str = "https://caldav.one.com/dav/"  # One.com CalDAV server
    
    # OneCom credentials
    EMAIL_ADDRESS: str
    EMAIL_PASSWORD: str
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./app.db"
    USER_DATABASE_URL: str = "sqlite:///./user.db"

    class Config:
        env_file = ".env"
        extra = "ignore"  # This will ignore extra fields in .env

settings = Settings()
