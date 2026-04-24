#!/usr/bin/env python3
import uvicorn
from config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=1,
        log_level="info",
    )
