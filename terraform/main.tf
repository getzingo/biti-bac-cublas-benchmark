resource "hyperstack_core_keypair" "mbp" {
  name        = "m_pubkey"
  environment = "default-CANADA-1"
  public_key  = var.ssh_pubkey
}


resource "hyperstack_core_virtual_machine" "bench" {
  name               = "bench"
  environment_name   = "default-CANADA-1"
  key_name           = "m_pubkey"
  image_name         = "Ubuntu Server 24.04 LTS R535 CUDA 12.2"
  flavor_name        = "n3-A100x1"
  user_data          = ""
  assign_floating_ip = true
}

resource "hyperstack_core_virtual_machine_sg_rule" "ssh_access" {
  virtual_machine_id = hyperstack_core_virtual_machine.bench.id

  direction        = "ingress"
  ethertype        = "IPv4"
  port_range_min   = 22
  port_range_max   = 22
  protocol         = "tcp"
  remote_ip_prefix = "0.0.0.0/0"
}