# celestial-syntax

## Team Members
- shaonirfan (Team Leader)
- Skalahuddin
- Bushra18433

## Mentor
- shakil-shahan

## Project Description
# Collaborative Study Platform

## Overview

This project aims to create a collaborative platform where students and teachers can manage and share study materials efficiently. The platform leverages AI to enhance the learning experience by organizing notes, generating quizzes, facilitating discussions, and creating opportunities for group study.

## Key Features

### 1. **Authentication & Role-Based Access Control**
- Separate authentication systems for students and teachers.
- Role-specific permissions:
  - Teachers can approve, edit, or annotate uploaded notes.
  - Students can upload and interact with approved content.

### 2. **Class Note Sharing**
- Students can upload notes as PDFs or images.
- OCR technology (e.g., Google Cloud Vision, Tesseract) is used to extract and organize text from uploaded files, making them searchable.

### 3. **AI-Driven Quiz Generation**
- Quizzes are generated from class notes using AI tools like OpenAI, Gemini, or Hugging Face.
- Questions align with the difficulty level and user roles (student/teacher).

### 4. **Chat with Class Notes**
- A chatbot feature enables users to query uploaded notes and slides from teacher.
- AI language models provide context-relevant responses.

### 5. **AI-Based Matchmaking for Study Groups**
- AI algorithms dynamically match students with similar interests or challenges for collaborative study sessions.
- Uses clustering or recommendation techniques for grouping.

### 6. **APIs and Custom Models**
- Integration with APIs (OpenAI, LangChain, etc.) for quiz generation, chatbot interaction, and note organization.
- Support for custom models to enhance features like note summarization or content validation.
- AI will generate questions based on previous year questions (eg. term final question).

## Technologies Used
- **OCR Tools:** Google Cloud Vision, Tesseract
- **AI Frameworks:** OpenAI, Gemini, Hugging Face, LangChain
- **Backend:** Custom APIs for data management and feature integration
- **Frontend:** Intuitive UI for role-based interactions

## Getting Started
1. Clone the repository
2. Install dependencies
3. Start development

## Development Guidelines
1. Create feature branches
2. Make small, focused commits
3. Write descriptive commit messages
4. Create pull requests for review

## Resources
- [Project Documentation](docs/)
- [Development Setup](docs/setup.md)
- [Contributing Guidelines](CONTRIBUTING.md)
