import threading
from datetime import datetime, timedelta

# Global variables: Store IP addresses and last access times
ip_last_access = {}
# Thread lock: Ensure thread-safe access to the dictionary
ip_lock = threading.Lock()

import gradio as gr
from functools import lru_cache
import random
import requests
import logging
import re
import config
import plotly.graph_objects as go
from typing import Dict
import json
from leaderboard import (
    get_current_leaderboard,
    update_leaderboard, 
    start_backup_thread, 
    get_leaderboard, 
    get_elo_leaderboard,
    ensure_elo_ratings_initialized,
)
import openai
from collections import Counter
from release_notes import get_release_notes_html

import os
import shutil
from datetime import datetime

# Directory to save uploaded images
UPLOAD_DIR = "uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Function to save the uploaded image with a unique filename
def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    save_path = os.path.join(UPLOAD_DIR, filename)
    image.save(save_path, format="PNG")
    return save_path

import io
import base64
from PIL import Image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

# Update the logging format to redact URLs
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress verbose HTTP request logging
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)

class RedactURLsFilter(logging.Filter):
    def filter(self, record):
        # Redact all URLs using regex pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        record.msg = re.sub(url_pattern, '[REDACTED_URL]', str(record.msg))
        
        # Remove HTTP status codes
        record.msg = re.sub(r'HTTP/\d\.\d \d+ \w+', '', record.msg)
        
        # Remove sensitive API references
        record.msg = record.msg.replace(config.API_URL, '[API]')
        record.msg = record.msg.replace(config.NEXTCLOUD_URL, '[CLOUD]')
        
        # Clean up residual artifacts
        record.msg = re.sub(r'\s+', ' ', record.msg).strip()
        record.msg = re.sub(r'("?) \1', '', record.msg)  # Remove empty quotes
        
        return True

# Apply the filter to all handlers
logger = logging.getLogger(__name__)
for handler in logging.root.handlers:
    handler.addFilter(RedactURLsFilter())

# Start the backup thread
start_backup_thread()

# Function to get available models (using predefined list)
def get_available_models():
    return [model[0] for model in config.get_approved_models()]

# Function to get recent opponents for a model
recent_opponents = {}

def update_recent_opponents(model_a, model_b):
    recent_opponents.setdefault(model_a, []).append(model_b)
    recent_opponents.setdefault(model_b, []).append(model_a)
    # Limit history to last 5 opponents
    recent_opponents[model_a] = recent_opponents[model_a][-5:]
    recent_opponents[model_b] = recent_opponents[model_b][-5:]

# API call to handle image inputs
# @lru_cache(maxsize=100)
def call_ollama_api(model, img_base64, text_prompt):
    client = openai.OpenAI(
        api_key=config.API_KEY,
        base_url=config.API_URL
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "low"}}
                    ]
                }
            ],
            timeout=180
        )
        content = response.choices[0].message.content
        return [
            {"role": "user", "content": "User input"},
            {"role": "assistant", "content": content}
        ]
    except Exception as e:
        return [
            {"role": "user", "content": "User input"},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]

# Generate responses using two randomly selected models
def get_battle_counts():
    leaderboard = get_current_leaderboard()
    battle_counts = Counter()
    for model, data in leaderboard.items():
        battle_counts[model] = data['wins'] + data['losses'] + data['ties']
    return battle_counts

def generate_responses(img_base64, text_prompt):
    available_models = get_available_models()
    if len(available_models) < 2:
        return [
            {"role": "user", "content": text_prompt},
            {"role": "assistant", "content": "Error: Not enough models available"}
        ], [
            {"role": "user", "content": text_prompt},
            {"role": "assistant", "content": "Error: Not enough models available"}
        ], None, None
    
    battle_counts = get_battle_counts()
    
    # Sort models by battle count (ascending)
    sorted_models = sorted(available_models, key=lambda m: battle_counts.get(m, 0))
    
    # Select the first model (least battles)
    model_a = sorted_models[0]
    
    # Filter out recent opponents for model_a
    potential_opponents = [m for m in sorted_models[1:] if m not in recent_opponents.get(model_a, [])]
    
    # If no potential opponents left, reset recent opponents for model_a
    if not potential_opponents:
        recent_opponents[model_a] = []
        potential_opponents = sorted_models[1:]
    
    # For the second model, use weighted random selection
    weights = [1 / (battle_counts.get(m, 1) + 1) for m in potential_opponents]
    model_b = random.choices(potential_opponents, weights=weights, k=1)[0]
    
    # Update recent opponents
    update_recent_opponents(model_a, model_b)
    
    # Get responses from both models
    response_a = call_ollama_api(model_a, img_base64, text_prompt)
    response_b = call_ollama_api(model_b, img_base64, text_prompt)
    
    # Return responses directly (already formatted correctly)
    return response_a, response_b, model_a, model_b

def battle_arena(image, text_prompt, request: gr.Request):

    # Rate limiting based on IP address
    current_time = datetime.now()
    ip = request.client.host  # 获取客户端 IP 地址
    print(f"Received request from IP: {ip}")

    with ip_lock:
        last_access = ip_last_access.get(ip)
        if last_access and current_time - last_access < timedelta(minutes=0.25):
            return (
                [], [], None, None,
                gr.update(value=[]),
                gr.update(value=[]),
                gr.update(interactive=False, value="Voting Disabled - Rate Limit Exceeded"),
                gr.update(interactive=False, value="Voting Disabled - Rate Limit Exceeded"),
                gr.update(interactive=False, visible=False),
                text_prompt,
                0,
                gr.update(visible=False),
                gr.update(value="⚠️ Warning ⚠️: You can only submit once every 15s", visible=True)
            )
        ip_last_access[ip] = current_time
    # Rate limit check passed, proceed with the battle

    img_base64 = image_to_base64(image)
    response_a, response_b, model_a, model_b = generate_responses(img_base64, text_prompt)
    
    # Check for API errors in responses
    if any("Error: Unable to get response from the model" in msg["content"]
           for msg in response_a + response_b 
           if msg["role"] == "assistant"):
        return (
            [], [], None, None,
            gr.update(value=[]),
            gr.update(value=[]),
            gr.update(interactive=False, value="Voting Disabled - API Error"),
            gr.update(interactive=False, value="Voting Disabled - API Error"),
            gr.update(interactive=False, visible=False),
            text_prompt,
            0,
            gr.update(visible=False),
            gr.update(value="Error: Unable to get response from the model", visible=True)
        )
    
    nickname_a = random.choice(config.model_nicknames)
    nickname_b = random.choice(config.model_nicknames)
    
    # The responses are already in the correct format, no need to reformat
    if random.choice([True, False]):
        return (
            response_a, response_b, model_a, model_b,
            gr.update(label=nickname_a, value=response_a),
            gr.update(label=nickname_b, value=response_b),
            gr.update(interactive=True, value=f"Vote for {nickname_a}"),
            gr.update(interactive=True, value=f"Vote for {nickname_b}"),
            gr.update(interactive=True, visible=True),
            text_prompt,
            0,
            gr.update(visible=False),
            gr.update(value="Ready for your vote! 🗳️", visible=True)
        )
    else:
        return (
            response_b, response_a, model_b, model_a,
            gr.update(label=nickname_a, value=response_b),
            gr.update(label=nickname_b, value=response_a),
            gr.update(interactive=True, value=f"Vote for {nickname_a}"),
            gr.update(interactive=True, value=f"Vote for {nickname_b}"),
            gr.update(interactive=True, visible=True),
            text_prompt,
            0,
            gr.update(visible=False),
            gr.update(value="Ready for your vote! 🗳️", visible=True)
        )

def record_vote(image, text_prompt, left_response, right_response, left_model, right_model, choice):
    # Check if outputs are generated
    if not left_response or not right_response or not left_model or not right_model:
        return (
            "Please generate responses before voting.", 
            gr.update(), 
            gr.update(interactive=False), 
            gr.update(interactive=False), 
            gr.update(visible=False), 
            gr.update()
        )
    saved_image_path = save_image(image)
    
    # winner = left_model if choice == "Left is better" else right_model
    # loser = right_model if choice == "Left is better" else left_model

    if choice == 'Left is better':
        winner, loser = left_model, right_model
        winner_response, loser_response = left_response[-1]['content'], right_response[-1]['content']
    else:
        winner, loser = right_model, left_model
        winner_response, loser_response = right_response[-1]['content'], left_response[-1]['content']
    
    # Update the leaderboard
    battle_results = update_leaderboard(winner, loser, saved_image_path, text_prompt, winner_response, loser_response, tie=False)
    
    result_message = f"""
    🎉 Vote recorded! You're awesome! 🌟
    🔵 In the left corner: {get_human_readable_name(left_model)}
    🔴 In the right corner: {get_human_readable_name(right_model)}
    🏆 And the champion you picked is... {get_human_readable_name(winner)}! 🥇
    """
    
    return (
        gr.update(value=result_message, visible=True),  # Show result as Markdown
        get_leaderboard(),                              # Update leaderboard
        get_elo_leaderboard(),                         # Update ELO leaderboard
        gr.update(interactive=False),                   # Disable left vote button
        gr.update(interactive=False),                   # Disable right vote button
        gr.update(interactive=False),                   # Disable tie button
        gr.update(visible=True)                         # Show model names
    )

# Deprecated
def get_leaderboard_chart():
    battle_results = get_current_leaderboard()
    
    # Calculate scores and sort results
    for model, results in battle_results.items():
        total_battles = results["wins"] + results["losses"] + results["ties"]
        if total_battles > 0:
            win_rate = results["wins"] / total_battles
            results["score"] = win_rate * (1 - 1 / (total_battles + 1))
        else:
            results["score"] = 0
    
    sorted_results = sorted(
        battle_results.items(), 
        key=lambda x: (x[1]["score"], x[1]["wins"] + x[1]["losses"] + x[1]["ties"]), 
        reverse=True
    )

    models = [get_human_readable_name(model) for model, _ in sorted_results]
    wins = [results["wins"] for _, results in sorted_results]
    losses = [results["losses"] for _, results in sorted_results]
    ties = [results["ties"] for _, results in sorted_results]
    scores = [results["score"] for _, results in sorted_results]

    fig = go.Figure()

    # Stacked Bar chart for Wins and Losses
    fig.add_trace(go.Bar(
        x=models,
        y=wins,
        name='Wins',
        marker_color='#22577a'
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=losses,
        name='Losses',
        marker_color='#38a3a5'
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=ties,
        name='Ties',
        marker_color='#57cc99'
    ))

    # Line chart for Scores
    fig.add_trace(go.Scatter(
        x=models,
        y=scores,
        name='Score',
        yaxis='y2',
        line=dict(color='#ff7f0e', width=2)
    ))

    # Update layout for full-width, increased height, and secondary y-axis
    fig.update_layout(
        title='Model Performance',
        xaxis_title='Models',
        yaxis_title='Number of Battles',
        yaxis2=dict(
            title='Score',
            overlaying='y',
            side='right'
        ),
        barmode='stack',
        height=800,
        width=1450,
        autosize=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    chart_data = fig.to_json()
    return fig

def new_battle():
    nickname_a = random.choice(config.model_nicknames)
    nickname_b = random.choice(config.model_nicknames)
    return (
        # "", # Reset prompt_input
        gr.update(value=None, label="Upload your image prompt"),  # Reset image input
        gr.update(value=[], label=nickname_a),  # Reset left Chatbot
        gr.update(value=[], label=nickname_b),  # Reset right Chatbot
        None,
        None,
        gr.update(interactive=False, value=f"Vote for {nickname_a}"),
        gr.update(interactive=False, value=f"Vote for {nickname_b}"),
        gr.update(interactive=False, visible=False),  # Reset Tie button
        gr.update(value="", visible=False),
        gr.update(),
        gr.update(visible=False),
        gr.update(),
        0  # Reset tie_count
    )

# Add this new function
def get_human_readable_name(model_name: str) -> str:
    model_dict = dict(config.get_approved_models())
    return model_dict.get(model_name, model_name)

# Add this new function to randomly select a prompt
def random_prompt():
    return random.choice(config.example_prompts)

# Modify the continue_conversation function
# Deprecated, the logic for ties has been modified
def continue_conversation(image, text_prompt, left_chat, right_chat, left_model, right_model, previous_prompt, tie_count):
    # Check if the prompt is empty or the same as the previous one
    # if not prompt or prompt == previous_prompt:
    #     prompt = random.choice(config.example_prompts)
    
    # Get responses (which are lists of messages)
    img_base64 = image_to_base64(image)
    left_response = call_ollama_api(left_model, img_base64, text_prompt)
    right_response = call_ollama_api(right_model, img_base64, text_prompt)
    
    # Append messages from the response lists
    left_chat.extend(left_response)
    right_chat.extend(right_response)
    
    tie_count += 1
    tie_button_state = gr.update(interactive=True) if tie_count < 3 else gr.update(interactive=False, value="Max ties reached. Please vote!")
    
    return (
        gr.update(value=left_chat),
        gr.update(value=right_chat),
        # gr.update(value=""),  # Clear the prompt input
        gr.update(value=image),  # Show the image input
        tie_button_state,
        text_prompt,  # Return the new prompt
        tie_count
    )

def record_tie(image, text_prompt, left_response, right_response, left_model, right_model):
    # save tie to leaderboard
    saved_image_path = save_image(image)
    left_response_text = left_response[-1]['content'] if left_response else "No response"
    right_response_text = right_response[-1]['content'] if right_response else "No response"
    battle_results = update_leaderboard(left_model, right_model, saved_image_path, text_prompt, left_response_text, right_response_text, tie=True)

    result_message = (
        f"🤝 Tie has been recorded!\n"
        f"🔵 {get_human_readable_name(left_model)}\n"
        f"🔴 {get_human_readable_name(right_model)}"
    )
    return (
        gr.update(value=result_message, visible=True),  # 显示平局
        get_leaderboard(),                              # 刷新 leaderboard
        get_elo_leaderboard(),                          # 刷新 elo
        gr.update(interactive=False),                   # 禁用左投票
        gr.update(interactive=False),                   # 禁用右投票
        gr.update(interactive=False),                   # 禁用 tie 按钮
        gr.update(visible=True)                         # 显示模型名
    )


def normalize_parameter_size(param_size: str) -> str:
    """Convert parameter size to billions (B) format."""
    try:
        # Remove any spaces and convert to uppercase for consistency
        param_size = param_size.replace(" ", "").upper()
        
        # Extract the number and unit
        if 'M' in param_size:
            # Convert millions to billions
            number = float(param_size.replace('M', '').replace(',', ''))
            return f"{number/1000:.2f}B"
        elif 'B' in param_size:
            # Already in billions, just format consistently
            number = float(param_size.replace('B', '').replace(',', ''))
            return f"{number:.2f}B"
        else:
            # If no unit or unrecognized format, try to convert the raw number
            number = float(param_size.replace(',', ''))
            if number >= 1000000000:
                return f"{number/1000000000:.2f}B"
            elif number >= 1000000:
                return f"{number/1000000000:.2f}B"
            else:
                return f"{number/1000000000:.2f}B"
    except:
        return param_size  # Return original if conversion fails

def load_latest_model_stats():
    """Load model stats from the model_stats.json file."""
    try:
        # Read directly from model_stats.json in root directory
        with open('model_stats.json', 'r') as f:
            stats = json.load(f)
            
        # Convert stats to table format
        table_data = []
        headers = ["Model", "Parameters", "Open Source", "Company"]
        
        for model in stats:
            if not model.get("success", False):  # Skip failed tests
                continue
                
            perf = model.get("performance", {})
            info = model.get("model_info", {})
            
            try:
                # # Format numeric values with 2 decimal places
                # model_size = float(info.get("size", 0))  # Get raw size
                # vram_gb = round(model_size/1024/1024/1024, 2)  # Convert to GB
                # tokens_per_sec = round(float(perf.get("tokens_per_second", 0)), 2)
                # gen_tokens_per_sec = round(float(perf.get("generation_tokens_per_second", 0)), 2)
                # total_tokens = perf.get("total_tokens", 0)
                # response_time = round(float(perf.get("response_time", 0)), 2)
                
                # # Normalize parameter size to billions format
                # param_size = normalize_parameter_size(info.get("parameter_size", "Unknown"))
                
                row = [
                    model.get("model_name", "Unknown"),      # String
                    info.get("parameter_size", "Unknown"),  # String, e.g., "7B", "13B"
                    info.get("opensource", "Unknown"),      # String
                    info.get("company", "Unknown"),          # String
                ]
                table_data.append(row)
            except Exception as row_error:
                logger.warning(f"Skipping model {model.get('model_name', 'Unknown')}: {str(row_error)}")
                continue
            
        if not table_data:
            return None, "No valid model stats found"
            
        # Sort by tokens per second (numerically)
        # table_data.sort(key=lambda x: float(x[5]) if isinstance(x[5], (int, float)) else 0, reverse=True)
        
        return headers, table_data
    except Exception as e:
        logger.error(f"Error in load_latest_model_stats: {str(e)}")
        return None, f"Error loading model stats: {str(e)}"

# Initialize Gradio Blocks
# start Gradio
with gr.Blocks(css="""
    #dice-button {
        min-height: 90px;
        font-size: 35px;
    }
    .sponsor-button {
        background-color: #30363D;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-weight: bold;
    }
    .sponsor-button:hover {
        background-color: #2D333B;
    }
""") as demo:
    gr.Markdown(config.ARENA_NAME)
    
    # Main description with sponsor button
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("""
                Step right up to the GeoArena!
                Watch as state-of-the-art AI models compete to pinpoint the location of each image with precision.

                Here, our contenders are challenged with diverse images from around the globe—urban scenes, landscapes, and more. Each round tests their ability to interpret visual clues and produce the most accurate guess.

                This is part of an ongoing research project. For inquiries or collaboration, please reach out to jia.pengyue@my.cityu.edu.hk
            """)
        # with gr.Column(scale=2):
        #     gr.Button(
        #         "Sponsor on GitHub",
        #         link="https://github.com/sponsors/k-mktr",
        #         elem_classes="sponsor-button"
        #     )
    
    # Instructions in an accordion
    with gr.Accordion("📖 How to Use", open=False):
        gr.Markdown("""
            1. To start the battle, go to the 'Battle Arena' tab.
            2. Upload your query image and Type your prompt into the text box.
            3. Click the "Generate Responses" button to view the models' responses.
            4. Cast your vote for the model that provided the better response.
            5. Check out the Leaderboard to see how models rank against each other.
        """)

    with gr.Accordion("📋 Disclaimer and Limitation of Liability", open=False):
        gr.Markdown("""
            1. GeoArena is an academic research platform developed to benchmark the geolocalization capabilities of large vision-language models (LVLMs) using real-world, user-contributed images and human preference signals. All data collected and displayed on the platform, including images and voting records, are used solely for research and evaluation purposes. Participation in the platform is entirely voluntary.
            2. By submitting images or interacting with the platform, users affirm that they have the right to upload the content and agree not to submit any personally identifiable information, copyrighted material without permission, or any content that violates local or international laws.
            3. The GeoArena team and affiliated institutions do not assume responsibility for the accuracy, legality, or content of user-submitted materials, while we implement safeguards to preserve user privacy and ensure ethical usage of the data.
            4. Furthermore, GeoArena does not provide any commercial geolocation service and should not be used for security, legal, or operational decision-making. All results, rankings, and model evaluations presented through the platform are provided “as is,” without warranties of any kind, either express or implied.
        """)
    
    # Leaderboard Tab (now first)
    with gr.Tab("Leaderboard"):
        gr.Markdown("""
        ### Bradley-Terry Leaderboard
        This leaderboard uses the Bradley-Terry model to rank models based on their win rates and number of battles.

        """)
        leaderboard = gr.Dataframe(
            headers=["#", "Model", "BT Score", "Wins", "Losses", "Ties", "Total Battles", "Win Rate"],
            row_count=10,
            col_count=8,
            interactive=True,
            label="Leaderboard"
        )
    
    # Battle Arena Tab (now second)
    with gr.Tab("Battle Arena"):
        with gr.Row():
            # prompt_input = gr.Textbox(
            #     label="Enter your prompt", 
            #     placeholder="Type your prompt here...",
            #     scale=20
            # )
            # random_prompt_btn = gr.Button("🎲", scale=1, elem_id="dice-button")
            prompt_input = gr.Image(label="Upload your image prompt", type="pil", scale=20, height=300) # fix height
        text_prompt = gr.Textbox(
            label='Text Prompt (You can also type a prompt here)',
            value = "You are an expert in image geolocalization. Given an image, provide the most likely location it was taken.",
            lines=2,
            interactive=True
        )
        gr.Markdown("<br>")
        
        # Add the random prompt button functionality
        # random_prompt_btn.click(
        #     random_prompt,
        #     outputs=prompt_input
        # )
        
        submit_btn = gr.Button("Generate Responses", variant="primary")
        
        with gr.Row():
            left_output = gr.Chatbot(label=random.choice(config.model_nicknames), type="messages")
            right_output = gr.Chatbot(label=random.choice(config.model_nicknames), type="messages")
        
        with gr.Row():
            left_vote_btn = gr.Button(f"Vote for {left_output.label}", interactive=False)
            tie_btn = gr.Button("Tie 🙈 Continue with a new prompt", interactive=False, visible=False)
            right_vote_btn = gr.Button(f"Vote for {right_output.label}", interactive=False)
        
        result = gr.Textbox(
            label="Status", 
            interactive=False, 
            value="Generate responses to start the battle! 🚀",
            visible=True  # Always visible
        )
        
        with gr.Row(visible=False) as model_names_row:
            left_model = gr.Textbox(label="🔵 Left Model", interactive=False)
            right_model = gr.Textbox(label="🔴 Right Model", interactive=False)
        
        previous_prompt = gr.State("")  # Add this line to store the previous prompt
        tie_count = gr.State(0)  # Add this line to keep track of tie count
        
        new_battle_btn = gr.Button("New Battle")
    
    # ELO Leaderboard Tab
    with gr.Tab("ELO Leaderboard"):
        gr.Markdown("""
        ### ELO Rating System
        This leaderboard uses the online ELO rating system.
        Initial ratings are 1000.
        """)
        elo_leaderboard = gr.Dataframe(
            headers=["#", "Model", "ELO Rating", "Wins", "Losses", "Ties", "Total Battles", "Win Rate"],
            row_count=10,
            col_count=8,
            interactive=True,
            label="ELO Leaderboard"
        )
    
    # Latest Updates Tab
    with gr.Tab("Latest Updates"):
        release_notes = gr.HTML(get_release_notes_html())
        refresh_notes_btn = gr.Button("Refresh Updates")
        
        refresh_notes_btn.click(
            get_release_notes_html,
            outputs=[release_notes]
        )
    
    # Model Stats Tab
    with gr.Tab("Model Stats"):
        gr.Markdown("""
        ### Model Performance Statistics
        
        This tab shows detailed information for each model.
        
        """)
        
        headers, table_data = load_latest_model_stats()
        if headers:
            model_stats_table = gr.Dataframe(
                headers=headers,
                value=table_data,
                row_count=len(table_data),
                col_count=len(headers),
                interactive=True,
                label="Model Performance Statistics"
            )
        else:
            gr.Markdown(f"⚠️ {table_data}")  # Show error message if loading failed
    
    # Define interactions
    submit_btn.click(
        battle_arena,
        inputs=[prompt_input, text_prompt],
        outputs=[
            left_output, right_output, left_model, right_model, 
            left_output, right_output, left_vote_btn, right_vote_btn,
            tie_btn, previous_prompt, tie_count, model_names_row, result
        ]
    )
    
    left_vote_btn.click(
        lambda *args: record_vote(*args, "Left is better"),
        inputs=[prompt_input, text_prompt, left_output, right_output, left_model, right_model],
        outputs=[result, leaderboard, elo_leaderboard, left_vote_btn, 
                 right_vote_btn, tie_btn, model_names_row]
    )
    
    right_vote_btn.click(
        lambda *args: record_vote(*args, "Right is better"),
        inputs=[prompt_input, text_prompt, left_output, right_output, left_model, right_model],
        outputs=[result, leaderboard, elo_leaderboard, left_vote_btn, 
                 right_vote_btn, tie_btn, model_names_row]
    )
    
    # tie_btn.click(
    #     continue_conversation,
    #     inputs=[prompt_input, text_prompt, left_output, right_output, left_model, right_model, previous_prompt, tie_count],
    #     outputs=[left_output, right_output, prompt_input, tie_btn, previous_prompt, tie_count]
    # )

    tie_btn.click(
        record_tie,
        inputs=[prompt_input, text_prompt, left_output, right_output, left_model, right_model],
        outputs=[result, leaderboard, elo_leaderboard, left_vote_btn, right_vote_btn, tie_btn, model_names_row]
    )

    
    new_battle_btn.click(
        new_battle,
        outputs=[prompt_input, left_output, right_output, left_model, 
                right_model, left_vote_btn, right_vote_btn, tie_btn,
                result, leaderboard, model_names_row, elo_leaderboard, tie_count]
    )
    
    # Update leaderboard on launch
    demo.load(get_leaderboard, outputs=leaderboard)
    demo.load(get_elo_leaderboard, outputs=elo_leaderboard)

if __name__ == "__main__":
    # Initialize ELO ratings before launching the app
    ensure_elo_ratings_initialized()
    # Start the model refresh thread
    config.start_model_refresh_thread()
    demo.launch(show_api=False, share=True)