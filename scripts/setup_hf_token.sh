#!/bin/bash

echo "========================================"
echo "Hugging Face Token Setup for GigaSpeech"
echo "========================================"
echo ""
echo "GigaSpeech is a gated dataset that requires authentication."
echo ""
echo "Steps to get access:"
echo "1. Create a Hugging Face account at https://huggingface.co/join"
echo "2. Go to https://huggingface.co/datasets/speechcolab/gigaspeech"
echo "3. Click 'Accept License' to accept the terms"
echo "4. Go to https://huggingface.co/settings/tokens"
echo "5. Create a new token with 'read' permissions"
echo ""
echo "Then export your token before running training:"
echo "  export HUGGING_FACE_HUB_TOKEN='your_token_here'"
echo ""
echo "Or pass it when starting remote training:"
echo "  HUGGING_FACE_HUB_TOKEN='your_token' python scripts/start_remote_training.py <IP> <PORT>"
echo ""
echo "========================================"

# Check if token is already set
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo ""
    echo "✓ Token is already set in environment"
    exit 0
fi

# Check if user wants to set it now
echo ""
read -p "Do you want to set your token now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your Hugging Face token: " token
    echo ""
    echo "Add this to your shell configuration (~/.bashrc or ~/.zshrc):"
    echo "export HUGGING_FACE_HUB_TOKEN='$token'"
    echo ""
    echo "Or run this command to use it for the current session:"
    echo "export HUGGING_FACE_HUB_TOKEN='$token'"
fi