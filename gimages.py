import tkinter as tk
from tkinter import messagebox
from google_images_download import google_images_download

def download_images():
    query = query_entry.get()  
    num_images = num_entry.get()  
    if not query or not num_images.isdigit():
        messagebox.showerror("Invalid Input", "Please enter a valid query and number of images")
        return
    
    try:
        num_images = int(num_images) 
        response = google_images_download.googleimagesdownload()
        arguments = {
            "keywords": query,
            "limit": num_images,
            "print_urls": True,
            "output_directory": "downloaded_images"
        }
        response.download(arguments)
        messagebox.showinfo("Success", f"Downloaded {num_images} images for query '{query}'")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Google Image Downloader")
root.geometry("400x200")

query_label = tk.Label(root, text="Search Query:")
query_label.pack(pady=10)
query_entry = tk.Entry(root, width=40)
query_entry.pack(pady=5)

num_label = tk.Label(root, text="Number of Images:")
num_label.pack(pady=10)
num_entry = tk.Entry(root, width=10)
num_entry.pack(pady=5)


download_button = tk.Button(root, text="Download Images", command=download_images)
download_button.pack(pady=20)


root.mainloop()