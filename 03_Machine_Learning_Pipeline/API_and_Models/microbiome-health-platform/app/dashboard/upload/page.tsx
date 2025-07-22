"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-provider"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useToast } from "@/hooks/use-toast"
import { FileUp, Upload } from "lucide-react"

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [description, setDescription] = useState("")
  const [isUploading, setIsUploading] = useState(false)
  const { user } = useAuth()
  const router = useRouter()
  const { toast } = useToast()
  const supabase = createClient()

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!file) {
      toast({
        title: "Error",
        description: "Please select a file to upload",
        variant: "destructive",
      })
      return
    }

    if (!user) {
      toast({
        title: "Error",
        description: "You must be logged in to upload files",
        variant: "destructive",
      })
      return
    }

    setIsUploading(true)

    try {
      // Upload file to Supabase Storage
      const fileExt = file.name.split(".").pop()
      const fileName = `${user.id}-${Math.random().toString(36).substring(2)}.${fileExt}`
      const filePath = `microbiome/${fileName}`

      const { error: uploadError } = await supabase.storage.from("uploads").upload(filePath, file)

      if (uploadError) {
        throw uploadError
      }

      // Get the public URL
      const {
        data: { publicUrl },
      } = supabase.storage.from("uploads").getPublicUrl(filePath)

      // Save metadata to database
      const { error: dbError } = await supabase.from("uploads").insert([
        {
          user_id: user.id,
          file_name: file.name,
          file_path: filePath,
          file_url: publicUrl,
          description,
          file_type: file.type,
          file_size: file.size,
        },
      ])

      if (dbError) {
        throw dbError
      }

      toast({
        title: "Success",
        description: "File uploaded successfully",
      })

      // Redirect to dashboard
      router.push("/dashboard")
      router.refresh()
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "An error occurred during upload",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Upload Microbiome Data</h1>
        <p className="text-muted-foreground">Upload your microbiome data files for analysis</p>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Upload File</CardTitle>
            <CardDescription>Upload your microbiome data file in CSV or TSV format</CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="file">File</Label>
                <div className="border-2 border-dashed rounded-md p-6 flex flex-col items-center justify-center">
                  {file ? (
                    <div className="text-center">
                      <FileUp className="h-10 w-10 text-muted-foreground mx-auto mb-2" />
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                      <Button variant="ghost" size="sm" className="mt-2" onClick={() => setFile(null)}>
                        Change file
                      </Button>
                    </div>
                  ) : (
                    <>
                      <Upload className="h-10 w-10 text-muted-foreground mb-2" />
                      <p className="mb-2 text-sm font-semibold">Drag and drop your file here or click to browse</p>
                      <p className="text-xs text-muted-foreground">Supported formats: CSV, TSV (Max 10MB)</p>
                      <Input id="file" type="file" accept=".csv,.tsv" className="hidden" onChange={handleFileChange} />
                      <Button
                        variant="outline"
                        className="mt-4"
                        onClick={() => document.getElementById("file")?.click()}
                      >
                        Browse Files
                      </Button>
                    </>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Enter a description for this data file"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" disabled={!file || isUploading}>
                {isUploading ? "Uploading..." : "Upload"}
              </Button>
            </CardFooter>
          </form>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Upload Guidelines</CardTitle>
            <CardDescription>Follow these guidelines for accurate analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <h3 className="font-medium">File Format</h3>
              <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                <li>CSV or TSV files only</li>
                <li>Maximum file size: 10MB</li>
                <li>First row should contain column headers</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">Required Columns</h3>
              <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                <li>Taxonomy (genus or species level)</li>
                <li>Abundance values (relative or absolute)</li>
                <li>Sample metadata (if available)</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium">Data Processing</h3>
              <p className="text-sm text-muted-foreground">
                After upload, your data will be processed using our analysis pipeline. This may take a few minutes.
                You'll receive a notification when the analysis is complete.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
