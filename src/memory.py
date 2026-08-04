"""
Conversation memory for the RAG assistant.
"""

from langchain.memory import ConversationBufferMemory


class ConversationMemory:

    def __init__(self):
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
        )

    def load_history(self):
        """
        Return previous conversation.
        """
        return self.memory.load_memory_variables({})

    def save(self, question, answer):
        """
        Save one interaction.
        """
        self.memory.save_context(
            {"question": question},
            {"answer": answer},
        )

    def clear(self):
        """
        Clear the conversation history.
        """
        self.memory.clear()