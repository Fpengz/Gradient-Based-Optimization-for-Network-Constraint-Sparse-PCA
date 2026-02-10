#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
import json

def get_bundle_id():
    """Detects the bundle ID of the current terminal emulator."""
    term_program = os.environ.get('TERM_PROGRAM', '')
    if term_program == 'Apple_Terminal':
        return 'com.apple.Terminal'
    elif term_program == 'iTerm.app':
        return 'com.googlecode.iterm2'
    elif term_program == 'vscode':
        return 'com.microsoft.VSCode'
    elif term_program == 'Hyper':
        return 'co.zeit.hyper'
    return None

def send_notification(title, message, sound, bundle_id):
    """Sends a notification using terminal-notifier (if available) or osascript."""
    
    # Check for terminal-notifier
    has_terminal_notifier = False
    try:
        subprocess.run(['terminal-notifier', '-help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        has_terminal_notifier = True
    except FileNotFoundError:
        pass

    if has_terminal_notifier and bundle_id:
        cmd = [
            'terminal-notifier',
            '-title', title,
            '-message', message,
            '-sound', sound,
            '-activate', bundle_id
        ]
        subprocess.run(cmd)
    else:
        # Fallback to osascript
        # Note: 'sound name' in AppleScript works with system sounds.
        applescript = f'display notification "{message}" with title "{title}" sound name "{sound}"'
        subprocess.run(['osascript', '-e', applescript])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--event', choices=['notification', 'done'], required=True, help="Type of event")
    args, unknown = parser.parse_known_args()

    # Read stdin (hook protocol) - though we might not use the content, we should consume it
    # to avoid broken pipes if the CLI writes to it.
    try:
        if not sys.stdin.isatty():
            input_data = sys.stdin.read()
            # Optional: Log input_data for debugging
    except Exception:
        pass

    bundle_id = get_bundle_id()
    
    if args.event == 'notification':
        # This is for "Permission needed" or system alerts
        send_notification(
            title="Gemini CLI",
            message="Action Required: Permission Needed",
            sound="Glass",
            bundle_id=bundle_id
        )
    elif args.event == 'done':
        # This is for "Task done"
        send_notification(
            title="Gemini CLI",
            message="Task Completed",
            sound="Hero",
            bundle_id=bundle_id
        )

if __name__ == "__main__":
    main()
