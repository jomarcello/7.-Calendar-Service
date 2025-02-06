from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
import base64
from openai import AsyncOpenAI
import os
import logging
from datetime import datetime
import aiohttp
import json
import redis
import asyncio
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from playwright_stealth import stealth_async  # Voeg stealth toe
import random
from fastapi.responses import HTMLResponse

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Calendar Service",
    description="Service for managing economic calendar data",
    version="1.0.0"
)

# Initialize OpenAI client met de juiste key
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=60.0
)

# Initialize Redis client with betere error handling
try:
    redis_host = os.getenv("REDIS_HOST", "redis")  # Default naar service naam
    redis_port = os.getenv("REDIS_PORT", "6379")   # Als string
    
    logger.info(f"🔄 Initialiseren Redis connectie: {redis_host}:{redis_port}")
    
    redis_client = redis.Redis(
        host=redis_host,
        port=int(redis_port),  # Expliciet naar int converteren
        db=0,
        socket_connect_timeout=5,
        decode_responses=True
    )
    
    # Test de connectie
    redis_client.ping()
    logger.info("✅ Redis verbinding succesvol")
    
except Exception as e:
    logger.error(f"❌ Redis verbinding mislukt: {str(e)}")
    logger.error(f"Redis config: host={redis_host}, port={redis_port}")
    redis_client = None

# Initialize Gemini
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("✅ Google Gemini API geconfigureerd")
    else:
        logger.error("❌ Google API key niet gevonden")
except Exception as e:
    logger.error(f"❌ Gemini configuratie fout: {str(e)}")

# Gebruik environment variable voor Telegram URL
telegram_url = os.getenv("TELEGRAM_SERVICE_URL", "http://tradingview-telegram-service:5000")

async def get_calendar_data(raw=False):
    """Get calendar data with multiple fallback sources"""
    try:
        # Try Forex Factory first (onze primaire bron)
        logger.info("🔄 Trying Forex Factory...")
        data = await get_forex_factory_calendar()
        if data:
            logger.info("✅ Forex Factory data retrieved successfully")
            return data if raw else format_calendar_data(data)
            
        # Als Forex Factory faalt, probeer IG
        logger.info("🔄 Trying IG calendar as fallback...")
        data = await get_ig_calendar()
        if data:
            logger.info("✅ IG data retrieved successfully")
            return data
            
        raise Exception("All calendar sources failed")
        
    except Exception as e:
        logger.error(f"❌ All calendar sources failed: {str(e)}")
        raise

async def get_forex_factory_calendar():
    """Get calendar from Forex Factory using Firefox"""
    browser = None
    try:
        async with async_playwright() as p:
            logger.info("🚀 Starting Firefox for Forex Factory...")
            
            # Gebruik Firefox in headless mode
            browser = await p.firefox.launch(
                headless=True,
                firefox_user_prefs={
                    "media.navigator.enabled": False,
                    "media.peerconnection.enabled": False,
                    "privacy.trackingprotection.enabled": True,
                    "dom.webdriver.enabled": False,
                    "network.http.referer.spoofSource": True,
                    "privacy.resistFingerprinting": True
                }
            )
            
            logger.info("📝 Creating browser context...")
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                locale='en-US',
                timezone_id='America/New_York'
            )
            
            page = await context.new_page()
            await stealth_async(page)
            
            # Blokkeer onnodige resources
            await page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,css}", lambda route: route.abort())
            await page.route("**/{google-analytics,doubleclick,facebook,analytics}**", lambda route: route.abort())
            
            logger.info("🌐 Navigating to Forex Factory calendar...")
            try:
                # Direct naar kalender
                response = await page.goto(
                    'https://www.forexfactory.com/calendar?day=today',
                    wait_until='domcontentloaded',  # Verander naar domcontentloaded
                    timeout=30000
                )
                
                if not response or not response.ok:
                    logger.error(f"Failed to load page: {response.status if response else 'No response'}")
                    raise Exception("Failed to load Forex Factory")
                
                logger.info("✅ Calendar page loaded")
                
                # Wacht eerst op de tabel container
                await page.wait_for_selector(
                    '.calendar__table',
                    state='visible',
                    timeout=10000
                )
                
                # Evalueer direct de data zonder te wachten op alle rijen
                events = await page.evaluate("""() => {
                    const events = [];
                    const rows = document.querySelectorAll('.calendar__row:not(.calendar__row--day-breaker)');
                    
                    for (const row of rows) {
                        try {
                            const time = row.querySelector('.calendar__time')?.textContent.trim();
                            const currency = row.querySelector('.calendar__currency')?.textContent.trim();
                            const impact = row.querySelector('.calendar__impact')?.className.includes('high') ? 'high' : 
                                         row.querySelector('.calendar__impact')?.className.includes('medium') ? 'medium' : 'low';
                            const title = row.querySelector('.calendar__event')?.textContent.trim();
                            const actual = row.querySelector('.calendar__actual')?.textContent.trim();
                            const forecast = row.querySelector('.calendar__forecast')?.textContent.trim();
                            const previous = row.querySelector('.calendar__previous')?.textContent.trim();
                            
                            if (time && currency && title) {
                                events.push({
                                    "date": time,
                                    "country": currency,
                                    "title": title,
                                    "importance": impact,
                                    "actual": actual,
                                    "forecast": forecast,
                                    "previous": previous
                                });
                            }
                        } catch (e) {
                            console.error('Error parsing row:', e);
                        }
                    }
                    
                    return events;
                }""")
                
                if not events:
                    raise Exception("No calendar events found")
                    
                return events
                
            except Exception as e:
                logger.error(f"❌ Navigation error: {str(e)}")
                content = await page.content()
                with open('/tmp/error.html', 'w') as f:
                    f.write(content)
                logger.info("📝 Error HTML saved to /tmp/error.html")
                raise
            
    except Exception as e:
        logger.error(f"❌ Forex Factory calendar error: {str(e)}")
        return None
        
    finally:
        if browser:
            await browser.close()
            logger.info("🔒 Browser closed")

async def get_ig_calendar():
    """Get calendar from IG with improved error handling"""
    browser = None
    try:
        async with async_playwright() as p:
            logger.info("Starting Playwright browser...")
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-software-rasterizer',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )
            
            logger.info("Creating browser context...")
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                ignore_https_errors=True
            )
            
            # Blokkeer onnodige resources
            await context.route("**/*.{png,jpg,jpeg,gif,css,woff,woff2}", lambda route: route.abort())
            
            page = await context.new_page()
            
            # Kortere timeout en meer logging
            page.set_default_timeout(15000)  # 15 seconden timeout
            
            logger.info("Navigating to IG calendar...")
            try:
                response = await page.goto(
                    'https://www.ig.com/en/economic-calendar',
                    wait_until='domcontentloaded',
                    timeout=15000
                )
                
                if not response or not response.ok:
                    logger.error(f"Failed to load IG calendar: {response.status if response else 'No response'}")
                    # Probeer alternatieve URL
                    response = await page.goto(
                        'https://www.ig.com/uk/economic-calendar',
                        wait_until='domcontentloaded',
                        timeout=15000
                    )
                
                logger.info("Waiting for calendar table...")
                calendar = await page.wait_for_selector(
                    '.economic-calendar table, .economic-calendar-table, #economicCalendarTable',
                    state='visible',
                    timeout=15000
                )
                
                if not calendar:
                    raise Exception("Calendar element not found")
                    
                # Wacht kort voor JavaScript rendering
                await page.wait_for_timeout(2000)
                
                # Take screenshot
                screenshot = await calendar.screenshot()
                return base64.b64encode(screenshot).decode()
                
            except Exception as e:
                logger.error(f"Navigation error: {str(e)}")
                # Neem screenshot van error voor debugging
                await page.screenshot(path='/tmp/error.png')
                raise
            
    except Exception as e:
        logger.error(f"IG calendar error: {str(e)}", exc_info=True)
        return None
        
    finally:
        if browser:
            try:
                await browser.close()
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")

# Basis routes eerst
@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "ok", "service": "calendar"}

@app.get("/calendar")
async def get_calendar():
    """Get economic calendar events"""
    try:
        calendar_data = await get_calendar_data(raw=True)
        if not calendar_data:
            raise Exception("Could not fetch calendar data")
        return calendar_data
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar/impact/{level}")
async def get_calendar_by_impact(level: str):
    """Filter calendar events by impact level"""
    try:
        calendar_data = await get_calendar_data(raw=True)
        if not calendar_data:
            raise Exception("Could not fetch calendar data")
            
        filtered_data = [
            event for event in calendar_data 
            if event.get("importance", "").lower() == level.lower()
        ]
        return filtered_data
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calendar/currency/{code}")
async def get_calendar_by_currency(code: str):
    """Filter calendar events by currency code"""
    try:
        calendar_data = await get_calendar_data(raw=True)
        if not calendar_data:
            raise Exception("Could not fetch calendar data")
            
        filtered_data = [
            event for event in calendar_data 
            if event.get("country", "").lower() == code.lower()
        ]
        return filtered_data
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_calendar_with_vision(screenshot_base64):
    """Analyze calendar data with GPT-4"""
    try:
        if not screenshot_base64:
            raise ValueError("Calendar data is empty")
            
        logger.info("🤖 Starting GPT-4 analysis...")
        
        messages = [
            {
                "role": "system",
                "content": "Je bent een AI-assistent die economische gebeurtenissen moet formatteren in een duidelijke lijst."
            },
            {
                "role": "user",
                "content": f"Hier is de kalender data, formatteer deze als volgt:\n\n⏰ [TIJD] 🏳️ [LAND] - [EVENT]\nImpact: [HIGH=🔴/MEDIUM=🟡/LOW=🟢]\nActual: [indien beschikbaar]\nForecast: [indien beschikbaar]\nPrevious: [indien beschikbaar]\n\nData:\n{screenshot_base64}"
            }
        ]
        
        try:
            response = await client.chat.completions.create(
                model="gpt-4",  # Gebruik gewoon GPT-4
                messages=messages,
                max_tokens=1000
            )
            
            logger.info("✅ GPT-4 analysis complete")
            return f"📅 *ECONOMISCHE KALENDER*\n\n{response.choices[0].message.content}"
            
        except Exception as e:
            logger.error(f"❌ GPT-4 error: {str(e)}")
            return format_calendar_without_vision(screenshot_base64)
            
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return format_calendar_without_vision(screenshot_base64)

def format_calendar_without_vision(screenshot_base64):
    """Format calendar data without using vision API"""
    try:
        # Simpele fallback tekst als vision niet beschikbaar is
        return """📅 *ECONOMISCHE KALENDER*

⚠️ Vision API niet beschikbaar
Bekijk de kalender op: https://www.forexfactory.com/calendar

Tip: Upgrade je OpenAI API key voor volledige functionaliteit."""
        
    except Exception as e:
        logger.error(f"Error in fallback formatting: {str(e)}")
        return "Error formatting calendar data"

@app.get("/health")
async def health_check():
    """Uitgebreide health check"""
    status = {
        "service": "calendar",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "redis": "healthy" if redis_client else "unavailable",
            "openai": "configured" if os.getenv("OPENAI_API_KEY") else "missing"
        }
    }
    
    # Test Redis
    if redis_client:
        try:
            redis_client.ping()
        except Exception as e:
            status["dependencies"]["redis"] = f"error: {str(e)}"
    
    # Test OpenAI
    try:
        if not os.getenv("OPENAI_API_KEY"):
            status["status"] = "degraded"
            status["message"] = "OpenAI API key not configured"
    except Exception as e:
        status["status"] = "error"
        status["message"] = f"Configuration error: {str(e)}"
    
    return status

async def test_exchange_rates():
    """Test OpenExchangeRates API"""
    try:
        app_id = "63b7063f56c4462b9ec0c43c0f489fad"
        url = f"https://openexchangerates.org/api/latest.json?app_id={app_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Exchange rates fetched successfully: {json.dumps(data, indent=2)}")
                    return data
                else:
                    error = await response.text()
                    logger.error(f"Error fetching exchange rates: {error}")
                    return None
                    
    except Exception as e:
        logger.error(f"Exchange rates API error: {str(e)}")
        return None

@app.get("/rates")
async def get_rates():
    """Get latest exchange rates"""
    try:
        rates = await test_exchange_rates()
        if rates:
            return {
                "status": "success",
                "data": rates
            }
        else:
            return {
                "status": "error",
                "message": "Could not fetch exchange rates"
            }
    except Exception as e:
        logger.error(f"Error in rates fetch: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": "Could not fetch exchange rates"
        }

async def get_calendar_events():
    """Get economic calendar events via API"""
    try:
        url = "https://economic-calendar.tradingview.com/events"
        params = {
            "from": "2024-02-05",
            "to": "2024-02-05",
            "importance": ["high", "medium"]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    error = await response.text()
                    logger.error(f"Error fetching calendar: {error}")
                    return None
                    
    except Exception as e:
        logger.error(f"Calendar API error: {str(e)}")
        return None

@app.get("/test-rates")
async def test_rates():
    """Test OpenExchangeRates API"""
    try:
        app_id = "63b7063f56c4462b9ec0c43c0f489fad"
        url = f"https://openexchangerates.org/api/latest.json?app_id={app_id}"
        
        # Debug logging
        logger.info(f"Testing OpenExchangeRates API: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info("Response data:")
                    logger.info(json.dumps(data, indent=2))
                    return {
                        "status": "success",
                        "data": data
                    }
                else:
                    error = await response.text()
                    logger.error(f"Error response: {error}")
                    return {
                        "status": "error",
                        "message": error
                    }
                    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

async def refresh_calendar_cache():
    """Ververs de kalender cache handmatig"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"calendar_{today}"
        
        # Haal nieuwe data op
        screenshot = await get_forex_factory_calendar()
        if screenshot:
            events = await analyze_calendar_with_vision(screenshot)
        else:
            raise Exception("Could not get calendar screenshot")
        
        # Update cache
        if redis_client and events:
            redis_client.setex(cache_key, 300, events)
            
        return events
    except Exception as e:
        logger.error(f"Error refreshing calendar cache: {str(e)}")
        raise

@app.get("/refresh")
async def force_refresh():
    """Force refresh van de kalender data"""
    try:
        events = await refresh_calendar_cache()
        return {
            "status": "success",
            "message": "Calendar refreshed successfully",
            "data": events
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to refresh calendar: {str(e)}"
        }

# Background task voor periodieke updates
async def periodic_calendar_update():
    """Update de kalender elke 5 minuten"""
    while True:
        try:
            await refresh_calendar_cache()
            logger.info("Calendar cache updated successfully")
        except Exception as e:
            logger.error(f"Failed to update calendar cache: {str(e)}")
        finally:
            await asyncio.sleep(300)  # Wacht 5 minuten

@app.on_event("startup")
async def startup_event():
    """Start achtergrondtaken bij opstarten"""
    try:
        # Check OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OpenAI API key niet geconfigureerd!")
        else:
            logger.info("✅ OpenAI API key gevonden")
            # Test de API key
            try:
                models = await client.models.list()
                logger.info("✅ OpenAI API key is geldig")
                logger.info(f"Beschikbare modellen: {[model.id for model in models.data]}")
            except Exception as e:
                logger.error(f"❌ OpenAI API key test mislukt: {str(e)}")
        
        # Start periodic updates
        asyncio.create_task(periodic_calendar_update())
        logger.info("✅ Started periodic calendar updates")
        
    except Exception as e:
        logger.error(f"❌ Failed to start background tasks: {str(e)}")

def format_calendar_data(data, format="html"):
    """Format calendar data to readable text or HTML"""
    try:
        if isinstance(data, str) and data.startswith('data:image'):
            # Dit is een base64 screenshot
            return data
            
        # Anders is het TradingView API data
        if format == "html":
            # HTML template
            html = """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; }
                    .calendar { width: 100%; border-collapse: collapse; }
                    .calendar th, .calendar td { 
                        padding: 8px; 
                        border: 1px solid #ddd; 
                    }
                    .calendar th { 
                        background: #f5f5f5; 
                        text-align: left; 
                    }
                    .high-impact { color: red; }
                    .medium-impact { color: orange; }
                    .low-impact { color: green; }
                </style>
            </head>
            <body>
                <h1>📅 Economische Kalender</h1>
                <table class="calendar">
                    <tr>
                        <th>Tijd</th>
                        <th>Land</th>
                        <th>Event</th>
                        <th>Impact</th>
                        <th>Actual</th>
                        <th>Forecast</th>
                        <th>Previous</th>
                    </tr>
            """
            
            # Add events
            for event in data:
                impact_class = "high-impact" if event.get("importance") == "high" else \
                             "medium-impact" if event.get("importance") == "medium" else \
                             "low-impact"
                             
                impact_emoji = "🔴" if event.get("importance") == "high" else \
                             "🟡" if event.get("importance") == "medium" else "🟢"
                
                html += f"""
                    <tr>
                        <td>{event.get("date", "")}</td>
                        <td>{event.get("country", "")}</td>
                        <td>{event.get("title", "")}</td>
                        <td class="{impact_class}">{impact_emoji}</td>
                        <td>{event.get("actual", "")}</td>
                        <td>{event.get("forecast", "")}</td>
                        <td>{event.get("previous", "")}</td>
                    </tr>
                """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            return html
            
        else:
            # Markdown/text format (bestaande code)
            formatted_events = []
            for event in data:
                time = event.get("date", "")
                country = event.get("country", "")
                title = event.get("title", "")
                impact = "🔴" if event.get("importance") == "high" else "🟡"
                actual = event.get("actual", "")
                forecast = event.get("forecast", "")
                previous = event.get("previous", "")
                
                formatted_event = f"⏰ {time}\n🏳️ {country} - {title}\nImpact: {impact}\n"
                if actual: formatted_event += f"Actual: {actual}\n"
                if forecast: formatted_event += f"Forecast: {forecast}\n"
                if previous: formatted_event += f"Previous: {previous}\n"
                
                formatted_events.append(formatted_event)
            
            events_text = "\n\n".join(formatted_events)
            return f"📅 *ECONOMISCHE KALENDER*\n\n{events_text}"
        
    except Exception as e:
        logger.error(f"Error formatting calendar data: {str(e)}")
        return "Error formatting calendar data"

# Root endpoint voor health check
@app.get("/")
async def root():
    """Root endpoint voor health check"""
    return {"status": "healthy", "service": "calendar"}