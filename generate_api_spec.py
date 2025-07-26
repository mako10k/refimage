#!/usr/bin/env python3
"""
API specification generator for RefImage API.

This script generates OpenAPI specification for analysis and documentation.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from refimage.api import create_app
    from refimage.config import Settings
    
    # Create app with default settings
    settings = Settings()
    app = create_app(settings)
    
    # Generate OpenAPI schema
    openapi_schema = app.openapi()
    
    # Save to file
    output_file = Path(__file__).parent / "api_specification.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    
    print(f"✅ OpenAPI specification generated: {output_file}")
    print(f"📊 API Endpoints: {len(openapi_schema.get('paths', {}))}")
    
    # Print endpoint summary
    paths = openapi_schema.get('paths', {})
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                summary = details.get('summary', 'No summary')
                print(f"  {method.upper():6} {path:30} - {summary}")

except Exception as e:
    print(f"❌ Error generating API specification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
