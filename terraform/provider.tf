terraform {
  required_providers {
    hyperstack = {
      source = "NexGenCloud/hyperstack"
      version = "0.2.2-alpha"
    }
  }
}
provider "hyperstack" {}

variable "hs_token" {
  sensitive = true
}

variable "ssh_pubkey" {
  sensitive = true
}