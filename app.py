import gradio as gr
import numpy as np
import tensorflow as tf
import random
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# -----------------------
# Mood & Category Mapping
# -----------------------
moods = ["Relaxed", "Adventurous", "Romantic", "Cultural", "Spiritual", "Wild", "Solo", "Family", "Luxury", "Backpacking"]
categories = [
    "Beach", "Mountains", "City", "Historical", "Nature", "Desert", "Theme Park",
    "Luxury Resort", "Hostel", "Forest"
]

# Sample destinations for each category
destinations = {
    "Beach": ["Maldives", "Bali", "Goa"],
    "Mountains": ["Manali", "Swiss Alps", "Rocky Mountains"],
    "City": ["New York", "Tokyo", "Paris"],
    "Historical": ["Rome", "Kyoto", "Athens"],
    "Nature": ["Yosemite", "Banff", "Amazon Rainforest"],
    "Desert": ["Sahara", "Dubai", "Jaisalmer"],
    "Theme Park": ["Orlando", "Singapore", "Seoul Everland"],
    "Luxury Resort": ["Bora Bora", "Santorini", "Seychelles"],
    "Hostel": ["Berlin", "Lisbon", "Buenos Aires"],
    "Forest": ["Black Forest", "Ardennes", "Daintree"]
}

# Place descriptions, must-visit sights, food recommendations, shopping, and cultural insights
# Place descriptions, must-visit sights, food recommendations, shopping, and cultural insights
place_descriptions = {
    "Maldives": {
        "description": "🌴 The Maldives is a tropical paradise famous for its white sandy beaches, crystal-clear waters, and overwater bungalows. Perfect for a luxurious, romantic getaway.",
        "must_visit": ["Malé Atoll", "Ban'dos Island", "Dhigurah Island"],
        "food": ["🍴 Mas Huni (traditional Maldivian breakfast)", "🍲 Garudhiya (fish soup)", "🍖 Fihunu Mas (grilled fish)"],
        "shopping": ["🛍️ Local markets in Malé for handicrafts", "🛒 Fresh seafood at local fish markets"],
        "culture": "The Maldives has a rich Islamic culture, known for its hospitality and traditional Maldivian music and dance. The Bodu Beru is a traditional drumming performance you must experience!"
        
    },
    "Bali": {
        "description": "🏝️ Bali is known for its picturesque beaches, terraced rice paddies, and vibrant culture. It's the ultimate destination for relaxation, culture, and adventure.",
        "must_visit": ["Uluwatu Temple", "Sacred Monkey Forest Sanctuary", "Tanah Lot Temple"],
        "food": ["🍚 Nasi Goreng (fried rice)", "🍢 Satay (grilled skewers)", "🍍 Babi Guling (suckling pig)"],
        "shopping": ["🛍️ Markets in Ubud for handcrafted art", "🛒 Seminyak for high-end boutiques"],
        "culture": "Bali is steeped in Hindu culture. Traditional dance, intricate temple rituals, and colorful festivals like Nyepi are highlights of a Bali trip."
        
    },
    "Goa": {
        "description": "🌞 Goa offers golden beaches, vibrant nightlife, and a Portuguese-influenced culture. It's perfect for both beach lovers and history buffs.",
        "must_visit": ["Baga Beach", "Fort Aguada", "Basilica of Bom Jesus"],
        "food": ["🍚 Prawn Curry", "🥘 Xacuti (spicy chicken curry)", "🍹 Feni (local alcohol)"],
        "shopping": ["🛍️ Anjuna Flea Market for unique souvenirs", "🛒 Mapusa Market for fresh spices and handicrafts"],
        "culture": "Goa’s culture is a blend of Portuguese influence and traditional Indian heritage. The state is known for its lively music scene, especially trance and electronic beats."
       
    },
    "Manali": {
        "description": "🏔️ Nestled in the Himalayas, Manali is a popular hill station known for its breathtaking views, adventure activities, and Tibetan culture.",
        "must_visit": ["Solang Valley", "Rohtang Pass", "Hadimba Temple"],
        "food": ["🍴 Siddu (traditional Himachali bread)", "🍖 Chana Madra (chickpea curry)", "🍷 Apple Cider (locally produced)"],
        "shopping": ["🛍️ Mall Road for woolen clothes and local handicrafts", "🛒 Local markets for Kullu shawls and souvenirs"],
        "culture": "Manali is deeply influenced by Tibetan culture. The town celebrates the Tibetan New Year, and you can find monasteries and vibrant Tibetan markets."
    },
    "Swiss Alps": {
        "description": "⛰️ The Swiss Alps offer some of the most stunning mountain landscapes in the world. It’s a haven for skiing, hiking, and breathtaking panoramic views.",
        "must_visit": ["Matterhorn Mountain", "Lake Geneva", "Jungfraujoch - Top of Europe"],
        "food": ["🥨 Rösti (potato dish)", "🍖 Fondue", "🍫 Swiss Chocolate"],
        "shopping": ["🛍️ Zurich for luxury shopping", "🛒 Lucerne for Swiss watches and souvenirs"],
        "culture": "Switzerland’s culture is known for its precision and neutrality. The Swiss celebrate various local festivals, with a strong focus on music, art, and traditional Swiss alpine life."
    },
    "Rocky Mountains": {
        "description": "🏞️ The Rocky Mountains are a vast mountain range that stretches through Canada and the USA. Known for their rugged beauty, it’s perfect for nature lovers and adventure seekers.",
        "must_visit": ["Rocky Mountain National Park", "Banff National Park", "Lake Louise"],
        "food": ["🍖 Bison Steaks", "🍲 Elk Stew", "🍪 Maple Syrup Cookies"],
        "shopping": ["🛍️ Local art and souvenirs in Banff", "🛒 Jasper for handmade leather goods"],
        "culture": "The Rocky Mountains are a symbol of the Wild West and Canadian heritage. You’ll find Native American cultural influences in art, music, and festivals."
    },
    "New York": {
        "description": "🌆 The Big Apple, New York City, is a cultural melting pot. From iconic landmarks like Times Square to Broadway shows, it’s a city that never sleeps.",
        "must_visit": ["Statue of Liberty", "Central Park", "Empire State Building"],
        "food": ["🍔 New York-style Pizza", "🥯 Bagels with cream cheese", "🍦 Hot Dogs from street vendors"],
        "shopping": ["🛍️ 5th Avenue for luxury shopping", "🛒 SoHo for boutiques and local art"],
        "culture": "New York is a global hub for arts, culture, and finance. The city hosts diverse ethnic neighborhoods, iconic museums, and a thriving theater scene in Broadway."
    },
    "Tokyo": {
        "description": "🏙️ Tokyo blends modern innovation with traditional culture. From futuristic skyscrapers to historic temples, it’s an exciting city to explore.",
        "must_visit": ["Shibuya Crossing", "Meiji Shrine", "Tokyo Tower"],
        "food": ["🍣 Sushi", "🍜 Ramen", "🍡 Mochi (sweet rice cakes)"],
        "shopping": ["🛍️ Shinjuku for electronics and fashion", "🛒 Harajuku for street fashion and quirky items"],
        "culture": "Tokyo’s culture is a mix of modernity and tradition. It’s home to ancient shrines, tea ceremonies, and contemporary pop culture like anime and manga."
    },
    "Paris": {
        "description": "🇫🇷 Paris, the City of Lights, is famous for its art, architecture, and fashion. It’s the ultimate destination for romance, history, and gastronomy.",
        "must_visit": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
        "food": ["🥖 Baguette", "🧀 Croissant", "🍷 Wine and French Cheese"],
        "shopping": ["🛍️ Champs-Élysées for luxury shopping", "🛒 Le Marais for vintage and trendy boutiques"],
        "culture": "Paris is the cultural capital of the world. It has an eclectic mix of history, art, and literature. French culture celebrates fashion, wine, and fine dining."
    },
    "Rome": {
        "description": "🏛️ Rome is a city steeped in history, with ruins from ancient civilizations, iconic landmarks, and mouthwatering Italian cuisine.",
        "must_visit": ["Colosseum", "Vatican City", "Pantheon"],
        "food": ["🍝 Pasta alla Carbonara", "🍕 Margherita Pizza", "🍦 Gelato"],
        "shopping": ["🛍️ Via del Corso for high street fashion", "🛒 Piazza di Spagna for luxury brands"],
        "culture": "Rome is the heart of ancient Roman culture. With its historical monuments, piazzas, and vibrant street life, Rome offers a glimpse into one of the most influential civilizations in history."
    },
   
    "Amazon Rainforest": {
        "description": "🌿 One of the world’s most biodiverse places, the Amazon is a green ocean of life and mystery.",
        "must_visit": ["Manaus", "Meeting of Waters", "Canopy Walkways"],
        "food": ["🍲 Tacacá", "🍌 Fried plantains", "🐟 Grilled river fish"],
        "shopping": ["🛍️ Indigenous crafts", "🛒 Manaus markets"],
        "culture": "Home to hundreds of native tribes and spiritual traditions deeply tied to nature."
    },
    "Sahara": {
        "description": "🏜️ The Sahara is the world’s largest hot desert, offering vast dunes and starry nights.",
        "must_visit": ["Merzouga", "Erg Chebbi", "Timbuktu"],
        "food": ["🍖 Mechoui", "🍲 Couscous", "🍵 Mint tea"],
        "shopping": ["🛍️ Nomadic textiles", "🛒 Berber jewelry"],
        "culture": "Nomadic Berber culture, desert caravans, and ancient trade routes shape Sahara life."
    },
    "Dubai": {
        "description": "🌆 Dubai is a futuristic oasis with skyscrapers, luxury malls, and desert safaris.",
        "must_visit": ["Burj Khalifa", "Palm Jumeirah", "Desert Safari"],
        "food": ["🥘 Shawarma", "🍢 Grilled kebabs", "🍰 Dates and Arabic sweets"],
        "shopping": ["🛍️ Dubai Mall", "🛒 Gold Souk"],
        "culture": "Emirati culture blends tradition with ultramodern innovation and global luxury."
    },
    "Jaisalmer": {
        "description": "🏯 Jaisalmer is Rajasthan’s Golden City, known for its forts and endless sand dunes.",
        "must_visit": ["Jaisalmer Fort", "Sam Sand Dunes", "Patwon Ki Haveli"],
        "food": ["🍛 Ker Sangri", "🍲 Dal Baati Churma", "🥤 Makhania Lassi"],
        "shopping": ["🛍️ Mirror-embroidered textiles", "🛒 Camel leather goods"],
        "culture": "A rich Rajput legacy with folk music, puppet shows, and desert hospitality."
    },
    "Orlando": {
        "description": "🎢 Orlando is the world’s theme park capital — pure fun for kids and grownups alike!",
        "must_visit": ["Walt Disney World", "Universal Studios", "SeaWorld"],
        "food": ["🍗 Turkey legs", "🍩 Theme park treats", "🍕 American fast food"],
        "shopping": ["🛍️ Orlando Premium Outlets", "🛒 Disney merchandise shops"],
        "culture": "Pop culture paradise blending fantasy, nostalgia, and family fun."
    },
    "Singapore": {
        "description": "🌇 A futuristic city-state where cultures collide and street food reigns supreme.",
        "must_visit": ["Gardens by the Bay", "Sentosa", "Marina Bay Sands"],
        "food": ["🍜 Laksa", "🥟 Hainanese Chicken Rice", "🍢 Satay"],
        "shopping": ["🛍️ Orchard Road", "🛒 Bugis Street"],
        "culture": "A mix of Chinese, Malay, Indian, and Western influences — ultra-clean and high-tech!"
    },
    "Seoul Everland": {
        "description": "🎠 Everland is South Korea’s largest theme park, nestled in Seoul’s vibrant outskirts.",
        "must_visit": ["T Express coaster", "Zootopia", "Global Fair"],
        "food": ["🍜 Korean fried chicken", "🍧 Patbingsu", "🍢 Fish cake skewers"],
        "shopping": ["🛍️ K-pop merch", "🛒 Theme park souvenirs"],
        "culture": "Korean pop culture and innovation shine through colorful, family-friendly fun."
    },
    "Bora Bora": {
        "description": "🌺 Bora Bora is the ultimate honeymoon paradise with turquoise lagoons and luxury resorts.",
        "must_visit": ["Matira Beach", "Mount Otemanu", "Coral Gardens"],
        "food": ["🍣 Poisson Cru", "🍍 Tropical fruits", "🍹 Coconut cocktails"],
        "shopping": ["🛍️ Pearl shops", "🛒 Handicrafts in Vaitape"],
        "culture": "Polynesian traditions, dance, and relaxed island vibes define the culture."
    },
    "Santorini": {
        "description": "🏖️ Santorini is Greece’s most romantic island — white houses, blue domes, sunsets galore.",
        "must_visit": ["Oia Village", "Red Beach", "Akrotiri Ruins"],
        "food": ["🍆 Moussaka", "🐟 Grilled octopus", "🍷 Santorini wine"],
        "shopping": ["🛍️ Boutiques in Fira", "🛒 Hand-painted ceramics"],
        "culture": "A Cycladic gem known for its architecture, mythology, and serene pace."
    },
    "Seychelles": {
        "description": "🌴 Seychelles is a dreamlike island nation with exotic beaches and marine life.",
        "must_visit": ["Anse Lazio", "La Digue", "Vallée de Mai"],
        "food": ["🍛 Creole curries", "🐟 Grilled red snapper", "🥥 Coconut desserts"],
        "shopping": ["🛍️ Victoria market", "🛒 Local spices & shell jewelry"],
        "culture": "Creole traditions with African, French, and Indian roots, full of rhythm and color."
    },
    "Berlin": {
        "description": "🧱 Berlin is raw, historic, and modern — from the Berlin Wall to buzzing street life.",
        "must_visit": ["Berlin Wall", "Brandenburg Gate", "Museum Island"],
        "food": ["🌭 Currywurst", "🍞 Pretzels", "🍺 German beer"],
        "shopping": ["🛍️ Mauerpark Flea Market", "🛒 Berlin boutiques"],
        "culture": "Creative, rebellious, and artistic — Berlin pulses with alternative culture and history."
    },
    "Lisbon": {
        "description": "🌉 Lisbon is Portugal’s hilly seaside capital full of charm, tram rides, and history.",
        "must_visit": ["Belem Tower", "Alfama", "Jerónimos Monastery"],
        "food": ["🍮 Pastéis de Nata", "🐙 Polvo à Lagareiro", "🍷 Port wine"],
        "shopping": ["🛍️ LX Factory", "🛒 Local ceramics"],
        "culture": "Fado music, seafaring legacy, and soulful architecture define this coastal gem."
    },
    "Buenos Aires": {
        "description": "💃 The Paris of South America — tango, art, and European elegance meet Latin fire.",
        "must_visit": ["La Boca", "Recoleta Cemetery", "Plaza de Mayo"],
        "food": ["🥩 Asado (Argentine BBQ)", "🍷 Malbec wine", "🥟 Empanadas"],
        "shopping": ["🛍️ San Telmo Market", "🛒 Leather goods"],
        "culture": "Passionate tango, political art, and fútbol run deep in Porteño culture."
    },
    "Black Forest": {
        "description": "🌲 A fairytale forest in Germany, full of hiking trails, cuckoo clocks, and legends.",
        "must_visit": ["Triberg Waterfalls", "Baden-Baden", "Schiltach"],
        "food": ["🍰 Black Forest Cake", "🥩 Pork Schnitzel", "🍻 German lagers"],
        "shopping": ["🛍️ Cuckoo clocks", "🛒 Handmade woodcraft"],
        "culture": "Rooted in folklore and medieval tales — perfect for dreamy wanderers."
    },
    "Ardennes": {
        "description": "🏕️ Belgium’s green escape — thick forests, castles, and cozy countryside inns.",
        "must_visit": ["La Roche-en-Ardenne", "Bouillon Castle", "Semois Valley"],
        "food": ["🧀 Raclette", "🥘 Game meat stews", "🍻 Belgian beer"],
        "shopping": ["🛍️ Regional jams", "🛒 Handcrafted soaps"],
        "culture": "Slow living, medieval legends, and Belgian hospitality await in the Ardennes."
    },
    "Daintree": {
        "description": "🌳 The world’s oldest tropical rainforest, Daintree is a biodiversity hotspot in Australia.",
        "must_visit": ["Cape Tribulation", "Daintree Discovery Centre", "Mossman Gorge"],
        "food": ["🥭 Exotic fruits", "🐟 Barramundi", "🍦 Wattle seed ice cream"],
        "shopping": ["🛍️ Aboriginal art", "🛒 Local honey and oils"],
        "culture": "Home to Kuku Yalanji people — rich in ancient traditions and rainforest wisdom."
    }
}

    # Continue adding other destinations similarly...


# You can now use the same `suggest_destination` function as before and all cities will have detailed descriptions.

    
    # Additional destinations...


# -----------------------
# Model Training
# -----------------------
mood_to_index = {mood: i for i, mood in enumerate(moods)}
category_to_index = {cat: i for i, cat in enumerate(categories)}
index_to_category = {i: cat for cat, i in category_to_index.items()}

X_train = np.array([mood_to_index[m] for m in moods])
y_train = np.array([category_to_index[c] for c in categories])

model = Sequential([
    Embedding(input_dim=len(moods), output_dim=8, input_length=1),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(len(categories), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=0)

# -----------------------
# Authentication Function
# -----------------------
def check_login(username, password):
    return username == "admin" and password == "pass123"

# -----------------------
# Destination Suggestion
# -----------------------
def suggest_destination(mood, budget, duration, food, hotel, booking, vibe, interest):
    try:
        mood_encoded = np.array([mood_to_index[mood]]).reshape(1, -1)
        predicted_category_index = np.argmax(model.predict(mood_encoded, verbose=0))
        predicted_category = index_to_category[predicted_category_index]
        recommended_place = random.choice(destinations[predicted_category])
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        # Fetch place description, must-visit sights, and food
        place_info = place_descriptions.get(recommended_place, None)

        if not place_info:
            return f"Sorry, no details available for {recommended_place} at this time."

        description = place_info.get("description", "No description available.")
        must_visit = ", ".join(place_info.get("must_visit", ["No must-visit sights available."]))
        food = ", ".join(place_info.get("food", ["No food recommendations available."]))
        shopping = ", ".join(place_info.get("shopping", ["No shopping recommendations available."]))
        culture = place_info.get("culture", "No cultural insights available.")
        
        

        return f"""
    <div style='font-family:Arial; font-size:16px; color:black;'>
        <h3 style="background:#a7d5f2;padding:10px;border-radius:12px;">
            🌍 Your Travel Plan - Generated on {timestamp}
        </h3>
        <div style="background-color:#000;padding:10px;border-radius:12px;margin-bottom:10px; color:white;">
            <b>📌 Destination:</b> {recommended_place}<br>
            <b>💰 Budget:</b> {budget} | <b>📅 Duration:</b> {duration}<br>
            <b>🍽 Food Preference:</b> {food}<br>
            <b>🏨 Hotel Type:</b> {hotel}<br>
            <b>🎫 Booking Platform:</b> {booking}<br>
            <b>🎵 Vibe Playlist:</b> {vibe}<br>
            <b>🎯 Interest:</b> {interest}<br><br>
            
            <b>📚 About {recommended_place}:</b><br>
            {description}<br><br>
            
            <b>🏝 Must-Visit Sights:</b><br>
            {must_visit}<br><br>
            
            <b>🍴 Recommended Food:</b><br>
            {food}<br><br>
            
            <b>🛍 Shopping Tips:</b><br>
            {shopping}<br><br>
            
            <b>🎭 Cultural Insights:</b><br>
            {culture}
        </div>
        <div style="background:#b2dfdb;padding:10px;border-radius:12px;">
            ✨ Enjoy your adventure and safe travels!
        </div>
    </div>
    """

    except Exception as e:
        return f"Error occurred: {str(e)}"

# -----------------------
# Gradio UI App
# -----------------------
with gr.Blocks(title="AI Travel Planner") as app:
    logged_in = gr.State(False)

    with gr.Column(visible=True) as login_page:
        gr.Markdown("### 🔐 Login to Your Travel Planner")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_status = gr.Textbox(label="Status", interactive=False)

    with gr.Column(visible=False) as dashboard:
        gr.Markdown("## 🌴 Travel Dashboard ✈")
        mood_input = gr.Dropdown(choices=moods, label="🌟 Mood")
        budget_input = gr.Radio(["Low", "Medium", "High"], label="💰 Budget")
        duration_input = gr.Radio(["Weekend", "1 Week", "2 Weeks", "1 Month"], label="📅 Trip Duration")
        food_input = gr.Dropdown(choices=["Vegetarian", "Non-Vegetarian", "Vegan", "Local Cuisine"], label="🍽 Food Preference")
        hotel_input = gr.Dropdown(choices=["Hostel", "3-Star", "5-Star", "Luxury Resort"], label="🏨 Hotel Type")
        booking_input = gr.Dropdown(choices=["Booking.com", "Airbnb", "Expedia"], label="🎫 Booking Platform")
        vibe_input = gr.Dropdown(choices=["Chill", "Upbeat", "Lo-Fi", "Classical", "Electronic"], label="🎵 Music Vibe")
        interest_input = gr.Dropdown(choices=["Hiking", "Beaches", "Museums", "Nightlife", "Wildlife", "Relaxation"], label="🎯 Main Interest")
        submit_btn = gr.Button("🚀 Generate Travel Plan")
        result_html = gr.HTML(label="✨ Your Travel Plan")

        submit_btn.click(
            fn=suggest_destination,
            inputs=[mood_input, budget_input, duration_input, food_input, hotel_input, booking_input, vibe_input, interest_input],
            outputs=result_html
        )

    def handle_login(user, pw):
        if check_login(user, pw):
            return gr.update(visible=False), gr.update(visible=True), "✅ Login successful! Welcome to your dashboard."
        else:
            return gr.update(), gr.update(), "❌ Invalid credentials. Please try again."

    login_btn.click(fn=handle_login, inputs=[username, password],
                    outputs=[login_page, dashboard, login_status])

app.launch()
