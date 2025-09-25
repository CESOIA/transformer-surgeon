# Message examples for BLIP inference testing
# BLIP VQA works best with direct questions about the image
# BLIP-2 can handle more complex prompts and conversations
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "What are in the image?"
            },
        ],
    },
    {
        "role": "user", 
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "What color are the objects?"
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "How many objects are there?"
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg", 
            },
            {
                "type": "text",
                "text": "What is this?"
            },
        ],
    },
    # BLIP-2 specific: More detailed prompts work better
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "Describe this image in detail, including colors, objects, and their arrangement."
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "Question: What type of fruit is shown in the image? Answer:"
            },
        ],
    },
    # Example for captioning model (no text)
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
        ],
    },
]
