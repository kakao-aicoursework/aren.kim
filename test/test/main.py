import pynecone as pc
from test import send_message

def chat_app():
    input_text = pc.state("")
    chat_history = pc.state([])

    @pc.component
    def send_button():
        def on_click():
            user_message = input_text.value
            if user_message:
                response = send_message(user_message)
                chat_history.update([f"You: {user_message}", f"AI: {response}"])
                input_text.set_value("")

        return pc.button("Send", on_click=on_click)

    @pc.component
    def chat_display():
        return pc.column([pc.text(message) for message in chat_history.value])

    @pc.component
    def chat_input():
        return pc.text_input(input_text, placeholder="Type your message...")

    return pc.column([chat_display, chat_input, send_button])

pc.run(chat_app)
