from flask import Flask, render_template, request, session, send_file, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
import logging
from datetime import datetime
import uuid
import os
import base64
from werkzeug.utils import secure_filename

# Import audio processing components
from audio_api import audio_bp, register_audio_socketio_handlers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(size_bytes):
    return round(size_bytes / (1024 * 1024), 2)

def get_file_icon(extension):
    icons = {
        'pdf': '📄', 'doc': '📝', 'docx': '📝', 'txt': '📝',
        'xls': '📊', 'xlsx': '📊', 'ppt': '📋', 'pptx': '📋',
        'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️',
        'mp3': '🎵', 'wav': '🎵', 'mp4': '🎬', 'avi': '🎬', 'mov': '🎬',
        'zip': '🗜️', 'rar': '🗜️', '7z': '🗜️'
    }
    return icons.get(extension.lower(), '📎')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
# Use threading instead of eventlet for better Python 3.13 compatibility
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx', 
    'ppt', 'pptx', 'zip', 'rar', '7z', 'mp3', 'mp4', 'avi', 'mov', 'wav'
}

# Register audio processing blueprint
app.register_blueprint(audio_bp)

# Register audio SocketIO handlers
register_audio_socketio_handlers(socketio)

# Store active rooms and users
active_rooms = {}
user_rooms = {}
user_media_states = {}  # Track audio/video states
room_chat_history = {}  # Store chat messages for each room
room_banned_users = {}  # Track banned users per room

# AI/ML features removed: server no longer contains AI enhancement classes or handlers.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/room/<room_id>')
def room(room_id): 
    username = request.args.get('username', '')
    return render_template('room.html', room_id=room_id, username=username)

@app.route('/webgl-test')
def webgl_test():
    return render_template('webgl-test.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    room_id = request.form.get('room_id')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not room_id:
        return jsonify({'error': 'Room ID required'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        
        # Create room-specific directory
        room_dir = os.path.join(app.config['UPLOAD_FOLDER'], room_id)
        os.makedirs(room_dir, exist_ok=True)
        
        file_path = os.path.join(room_dir, filename)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        file_info = {
            'filename': filename,
            'original_filename': file.filename,
            'size_mb': get_file_size_mb(file_size),
            'extension': file_extension,
            'icon': get_file_icon(file_extension),
            'upload_time': get_timestamp(),
            'file_path': f'/download_file/{room_id}/{filename}'
        }
        
        return jsonify({
            'success': True,
            'file_info': file_info
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/upload_voice', methods=['POST'])
def upload_voice():
    if 'voice' not in request.files:
        return jsonify({'error': 'No voice file provided'}), 400
    
    voice_file = request.files['voice']
    room_id = request.form.get('room_id')
    duration = request.form.get('duration', '0')
    
    if voice_file.filename == '':
        return jsonify({'error': 'No voice file selected'}), 400
    
    if not room_id:
        return jsonify({'error': 'Room ID required'}), 400
    
    try:
        # Create secure filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + secure_filename('voice_message.webm')
        
        # Create room-specific directory
        room_dir = os.path.join(app.config['UPLOAD_FOLDER'], room_id, 'voices')
        os.makedirs(room_dir, exist_ok=True)
        
        file_path = os.path.join(room_dir, filename)
        voice_file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        
        voice_info = {
            'filename': filename,
            'size_mb': get_file_size_mb(file_size),
            'duration': int(duration),
            'upload_time': get_timestamp(),
            'voice_path': f'/download_voice/{room_id}/{filename}'
        }
        
        return jsonify({
            'success': True,
            'voice_info': voice_info
        })
        
    except Exception as e:
        logger.error(f"Error uploading voice message: {e}")
        return jsonify({'error': 'Failed to upload voice message'}), 500

@app.route('/download_voice/<room_id>/<filename>')
def download_voice(room_id, filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, 'voices', filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='audio/webm')
        else:
            return "Voice file not found", 404
    except Exception as e:
        logger.error(f"Error downloading voice file: {e}")
        return "Error downloading voice file", 500

@app.route('/download_file/<room_id>/<filename>')
def download_file(room_id, filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], room_id, filename)
        if os.path.exists(file_path):
            # Get original filename (remove timestamp prefix)
            original_filename = '_'.join(filename.split('_')[2:])
            return send_file(file_path, as_attachment=True, download_name=original_filename)
        else:
            return "File not found", 404
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return "Error downloading file", 500

@socketio.on('connect')
def on_connect():
    # Generate unique session ID for audio processing
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    logger.info(f"User {request.sid} connected with session {session['session_id']}")
    emit('connected', {'data': 'Connected to server', 'session_id': session['session_id']})

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"User {request.sid} disconnected")
    # Clean up user from rooms
    if request.sid in user_rooms:
        room_id = user_rooms[request.sid]
        username = user_media_states.get(request.sid, {}).get('username', f'User-{request.sid[:6]}')
        
        leave_room(room_id)
        if room_id in active_rooms:
            if request.sid in active_rooms[room_id]:
                active_rooms[room_id].remove(request.sid)
                # Notify other users in the room
                emit('user_left', {'user_id': request.sid}, room=room_id)
            if len(active_rooms[room_id]) == 0:
                del active_rooms[room_id]
                # Clean up chat history for empty rooms after some time
                if room_id in room_chat_history:
                    del room_chat_history[room_id]
        
        # Send system message about user leaving
        if room_id in room_chat_history:
            leave_message = {
                'type': 'system',
                'message': f'{username} left the room',
                'timestamp': get_timestamp(),
                'user_id': request.sid,
                'username': username
            }
            room_chat_history[room_id].append(leave_message)
            emit('new_message', leave_message, room=room_id)
        
        del user_rooms[request.sid]
    
    # Clean up media state
    if request.sid in user_media_states:
        del user_media_states[request.sid]

@socketio.on('join_room')
def on_join_room(data):
    room_id = data['room_id']
    user_id = request.sid
    username = data.get('username', f'User {user_id[:6].upper()}')
    
    logger.info(f"User {user_id} ({username}) attempting to join room {room_id}")
    
    # Check if user is banned from this room (check by username)
    if room_id in room_banned_users and username in room_banned_users[room_id]:
        logger.warning(f"Banned user {username} attempted to join room {room_id}")
        emit('you_are_banned', {'message': 'You are banned from this room'})
        return
    
    # Join the room
    join_room(room_id)
    user_rooms[user_id] = room_id
    
    # Initialize media state
    user_media_states[user_id] = {
        'audio': False,
        'video': False,
        'audio_muted': False,
        'video_muted': False,
        'username': username
    }
    
    # Initialize chat history for room if not exists
    if room_id not in room_chat_history:
        room_chat_history[room_id] = []
    
    # Initialize banned users list for room if not exists
    if room_id not in room_banned_users:
        room_banned_users[room_id] = set()
    
    # Add user to active rooms
    if room_id not in active_rooms:
        active_rooms[room_id] = []
    active_rooms[room_id].append(user_id)
    
    # Get list of existing users in the room (excluding the new user)
    existing_users = [uid for uid in active_rooms[room_id] if uid != user_id]
    
    # Notify the new user about existing users and their media states
    existing_users_with_states = []
    for uid in existing_users:
        user_state = user_media_states.get(uid, {})
        existing_users_with_states.append({
            'user_id': uid,
            'username': user_state.get('username', f'User {uid[:6].upper()}'),
            'media_state': user_state
        })
    
    emit('existing_users', {'users': existing_users_with_states})
    
    # Send chat history to new user
    emit('chat_history', {'messages': room_chat_history[room_id]})
    
    # Notify existing users about the new user
    emit('user_joined', {
        'user_id': user_id,
        'username': username,
        'media_state': user_media_states[user_id]
    }, room=room_id, include_self=False)
    
    # Send system message about user joining
    join_message = {
        'type': 'system',
        'message': f'{username} joined the room',
        'timestamp': get_timestamp(),
        'user_id': user_id,
        'username': username
    }
    room_chat_history[room_id].append(join_message)
    emit('new_message', join_message, room=room_id)
    
    logger.info(f"Room {room_id} now has {len(active_rooms[room_id])} users")

@socketio.on('leave_room')
def on_leave_room(data):
    room_id = data['room_id']
    user_id = request.sid
    
    logger.info(f"User {user_id} leaving room {room_id}")
    
    leave_room(room_id)
    
    if user_id in user_rooms:
        del user_rooms[user_id]
    
    if room_id in active_rooms and user_id in active_rooms[room_id]:
        active_rooms[room_id].remove(user_id)
        # Notify other users
        emit('user_left', {'user_id': user_id}, room=room_id)
        
        if len(active_rooms[room_id]) == 0:
            del active_rooms[room_id]

# WebRTC signaling events
@socketio.on('offer')
def on_offer(data):
    target_id = data['target']
    offer = data['offer']
    logger.info(f"Relaying offer from {request.sid} to {target_id}")
    emit('offer', {'offer': offer, 'sender': request.sid}, room=target_id)

@socketio.on('answer')
def on_answer(data):
    target_id = data['target']
    answer = data['answer']
    logger.info(f"Relaying answer from {request.sid} to {target_id}")
    emit('answer', {'answer': answer, 'sender': request.sid}, room=target_id)

@socketio.on('ice_candidate')
def on_ice_candidate(data):
    target_id = data['target']
    candidate = data['candidate']
    logger.info(f"Relaying ICE candidate from {request.sid} to {target_id}")
    emit('ice_candidate', {'candidate': candidate, 'sender': request.sid}, room=target_id)

@socketio.on('get_room_users')
def on_get_room_users(data):
    room_id = data['room_id']
    if room_id in active_rooms:
        users = active_rooms[room_id]
        emit('room_users', {'users': users})
    else:
        emit('room_users', {'users': []})

# Media state management
@socketio.on('update_media_state')
def on_update_media_state(data):
    user_id = request.sid
    if user_id in user_media_states:
        user_media_states[user_id].update(data)
        
        # Notify other users in the room about the media state change
        if user_id in user_rooms:
            room_id = user_rooms[user_id]
            emit('media_state_changed', {
                'user_id': user_id,
                'media_state': user_media_states[user_id]
            }, room=room_id, include_self=False)

@socketio.on('toggle_audio')
def on_toggle_audio(data):
    user_id = request.sid
    if user_id in user_media_states:
        user_media_states[user_id]['audio_muted'] = data.get('muted', False)
        
        if user_id in user_rooms:
            room_id = user_rooms[user_id]
            emit('user_audio_toggled', {
                'user_id': user_id,
                'muted': data.get('muted', False)
            }, room=room_id, include_self=False)

@socketio.on('toggle_video')
def on_toggle_video(data):
    user_id = request.sid
    if user_id in user_media_states:
        user_media_states[user_id]['video_muted'] = data.get('muted', False)
        
        if user_id in user_rooms:
            room_id = user_rooms[user_id]
            emit('user_video_toggled', {
                'user_id': user_id,
                'muted': data.get('muted', False)
            }, room=room_id, include_self=False)

# Chat message handlers
@socketio.on('send_message')
def on_send_message(data):
    user_id = request.sid
    if user_id not in user_rooms:
        return
    
    room_id = user_rooms[user_id]
    username = user_media_states.get(user_id, {}).get('username', f'User-{user_id[:6]}')
    
    message_data = {
        'type': 'user',
        'message': data.get('message', '').strip(),
        'timestamp': get_timestamp(),
        'user_id': user_id,
        'username': username
    }
    
    if message_data['message']:  # Only send non-empty messages
        # Store message in room history
        if room_id in room_chat_history:
            room_chat_history[room_id].append(message_data)
            
            # Keep only last 100 messages per room
            if len(room_chat_history[room_id]) > 100:
                room_chat_history[room_id] = room_chat_history[room_id][-100:]
        
        # Broadcast message to all users in the room
        emit('new_message', message_data, room=room_id)
        
        logger.info(f"Message from {username} in room {room_id}: {message_data['message']}")

@socketio.on('send_voice_message')
def on_send_voice_message(data):
    user_id = request.sid
    if user_id not in user_rooms:
        return
    
    room_id = user_rooms[user_id]
    username = user_media_states.get(user_id, {}).get('username', f'User-{user_id[:6]}')
    
    voice_info = data.get('voice_info', {})
    
    message_data = {
        'type': 'voice',
        'voice_info': voice_info,
        'timestamp': get_timestamp(),
        'user_id': user_id,
        'username': username
    }
    
    # Store message in room history
    if room_id in room_chat_history:
        room_chat_history[room_id].append(message_data)
        
        # Keep only last 100 messages per room
        if len(room_chat_history[room_id]) > 100:
            room_chat_history[room_id] = room_chat_history[room_id][-100:]
    
    # Broadcast voice message to all users in the room
    emit('new_message', message_data, room=room_id)
    
    logger.info(f"Voice message sent by {username} in room {room_id}: {voice_info.get('duration', 0)}s")

@socketio.on('send_file_message')
def on_send_file_message(data):
    user_id = request.sid
    if user_id not in user_rooms:
        return
    
    room_id = user_rooms[user_id]
    username = user_media_states.get(user_id, {}).get('username', f'User-{user_id[:6]}')
    
    file_info = data.get('file_info', {})
    
    message_data = {
        'type': 'file',
        'file_info': file_info,
        'timestamp': get_timestamp(),
        'user_id': user_id,
        'username': username
    }
    
    # Store message in room history
    if room_id in room_chat_history:
        room_chat_history[room_id].append(message_data)
        
        # Keep only last 100 messages per room
        if len(room_chat_history[room_id]) > 100:
            room_chat_history[room_id] = room_chat_history[room_id][-100:]
    
    # Broadcast file message to all users in the room
    emit('new_message', message_data, room=room_id)
    
    logger.info(f"File shared by {username} in room {room_id}: {file_info.get('original_filename', 'unknown')}")

@socketio.on('typing')
def on_typing(data):
    user_id = request.sid
    if user_id not in user_rooms:
        return
    
    room_id = user_rooms[user_id]
    username = user_media_states.get(user_id, {}).get('username', f'User-{user_id[:6]}')
    
    # Broadcast typing indicator to other users in the room
    emit('user_typing', {
        'user_id': user_id,
        'username': username,
        'is_typing': data.get('is_typing', False)
    }, room=room_id, include_self=False)

@socketio.on('stop_typing')
def on_stop_typing():
    user_id = request.sid
    if user_id not in user_rooms:
        return
    
    room_id = user_rooms[user_id]
    username = user_media_states.get(user_id, {}).get('username', f'User-{user_id[:6]}')
    
    # Broadcast stop typing to other users in the room
    emit('user_typing', {
        'user_id': user_id,
        'username': username,
        'is_typing': False
    }, room=room_id, include_self=False)

# AI/ML handlers removed.

# Ban management handlers
@socketio.on('ban_user')
def on_ban_user(data):
    banner_id = request.sid
    banned_user_id = data.get('user_id')
    
    if not banned_user_id or banner_id not in user_rooms:
        return
    
    room_id = user_rooms[banner_id]
    
    # Get the banned user's username
    banned_username = user_media_states.get(banned_user_id, {}).get('username', f'User {banned_user_id[:6].upper()}')
    
    # Initialize banned users list for room if not exists
    if room_id not in room_banned_users:
        room_banned_users[room_id] = set()
    
    # Add username to banned list (not socket ID)
    room_banned_users[room_id].add(banned_username)
    
    # If the banned user is currently in the room, kick them out
    if banned_user_id in user_rooms and user_rooms[banned_user_id] == room_id:
        # Notify the banned user they are being kicked
        emit('you_are_banned', {'message': 'You have been banned from this room'}, room=banned_user_id)
        
        # Remove user from room
        leave_room(room_id, sid=banned_user_id)
        
        # Clean up user data
        if banned_user_id in user_rooms:
            del user_rooms[banned_user_id]
        if banned_user_id in user_media_states:
            del user_media_states[banned_user_id]
        if room_id in active_rooms and banned_user_id in active_rooms[room_id]:
            active_rooms[room_id].remove(banned_user_id)
        
        # Notify other users in the room
        emit('user_left', {
            'user_id': banned_user_id,
            'username': banned_username,
            'reason': 'banned'
        }, room=room_id)
    
    logger.info(f"User {banned_username} banned from room {room_id} by {banner_id}")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)