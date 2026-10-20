defmodule EMLX.ToBinaryTest do
  use EMLX.Case, async: true

  import Bitwise

  test "to_binary for contiguous tensor" do
    t = Nx.iota({4, 3}, type: :f32, backend: EMLX.Backend)
    ref = Nx.iota({4, 3}, type: :f32, backend: Nx.BinaryBackend)
    assert Nx.to_binary(t) == Nx.to_binary(ref)
  end

  test "to_binary for large contiguous tensor" do
    t = Nx.iota({64, 1536}, type: :f32, backend: EMLX.Backend)
    ref = Nx.iota({64, 1536}, type: :f32, backend: Nx.BinaryBackend)
    assert Nx.to_binary(t) == Nx.to_binary(ref)
  end

  test "to_binary for transposed (non-contiguous) tensor" do
    t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend) |> Nx.transpose()
    ref = Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.transpose()
    assert Nx.to_binary(t) == Nx.to_binary(ref)
  end

  test "to_binary with limit for contiguous tensor" do
    t = Nx.iota({100}, type: :f32, backend: EMLX.Backend)
    ref = Nx.iota({100}, type: :f32, backend: Nx.BinaryBackend)
    assert Nx.to_binary(t, limit: 10) == Nx.to_binary(ref, limit: 10)
  end

  # Refcount safety: binary must remain valid after the tensor term is dropped and GC'd.
  # A segfault here means the resource binary does not hold the MLX buffer alive.
  test "binary remains readable after original tensor is GC'd" do
    test_pid = self()

    spawn(fn ->
      t = Nx.iota({4}, type: :f32, backend: EMLX.Backend)

      # Creates the resource binary, which should pin the MLX buffer
      dev_ref = EMLX.Backend.from_nx(t)
      bin = EMLX.to_blob(dev_ref)

      # t goes out of scope here
      send(test_pid, {:bin, bin})
    end)

    assert_receive {:bin, bin}, 1000

    :erlang.garbage_collect()
    assert byte_size(bin) == 4 * 4
    # Force a copy to verify every byte is readable — would segfault on UAF
    assert :binary.copy(bin) == bin
  end

  # Non-contiguous refcount safety: same test via the ct resource path.
  test "non-contiguous binary remains readable after original tensor is GC'd" do
    test_pid = self()

    spawn(fn ->
      t = Nx.iota({3, 4}, type: :f32, backend: EMLX.Backend) |> Nx.transpose()

      # Creates the resource binary, which should pin the MLX buffer
      dev_ref = EMLX.Backend.from_nx(t)
      bin = EMLX.to_blob(dev_ref)

      send(test_pid, {:bin, bin})
    end)

    assert_receive {:bin, bin}, 1000

    :erlang.garbage_collect()
    assert byte_size(bin) == 3 * 4 * 4
    assert :binary.copy(bin) == bin
  end

  test "f8 round-trips every non-NaN value" do
    codes = for byte <- 0..255, (byte &&& 0x7F) <= 0x7C, into: <<>>, do: <<byte>>
    t = Nx.from_binary(codes, {:f, 8}, backend: EMLX.Backend)
    assert Nx.to_binary(t) == codes
  end

  test "f8 keeps NaN" do
    f8_nans = Nx.from_binary(<<0x7E, 0xFE>>, {:f, 8}, backend: EMLX.Backend)
    assert f8_nans |> Nx.is_nan() |> Nx.to_flat_list() == [1, 1]

    f16_nans =
      Nx.from_binary(
        <<0x7C01::16-native, 0x7FFF::16-native, 0xFFFF::16-native>>,
        {:f, 16},
        backend: EMLX.Backend
      )

    assert f16_nans |> Nx.as_type(:f8) |> Nx.to_binary() == <<0x7E, 0x7E, 0xFE>>
  end

  test "f8 rounds to nearest, ties to even" do
    decode = fn code ->
      <<value::float-big-16>> = <<code, 0>>
      value
    end

    cases =
      for code <- 0..0x7A, sign <- [0, 0x80] do
        low = decode.(sign ||| code)
        high = decode.(sign ||| code + 1)
        quarter = (high - low) / 4
        even = sign ||| code + rem(code, 2)

        [
          {low, sign ||| code},
          {low + quarter, sign ||| code},
          {low + 2 * quarter, even},
          {high - quarter, sign ||| code + 1}
        ]
      end

    {values, expected} = cases |> List.flatten() |> Enum.unzip()

    t =
      values
      |> Nx.tensor(type: :f16, backend: EMLX.Backend)
      |> Nx.as_type(:f8)

    assert Nx.to_binary(t) == :binary.list_to_bin(expected)
  end

  test "f8 rounds past the largest finite value to infinity" do
    t =
      [61_408.0, 61_440.0, -61_440.0]
      |> Nx.tensor(type: :f16, backend: EMLX.Backend)
      |> Nx.as_type(:f8)

    assert Nx.to_binary(t) == <<0x7B, 0x7C, 0xFC>>
  end

  test "f8 results are finite" do
    t =
      [0.875, 3.5]
      |> Nx.tensor(type: :f8, backend: EMLX.Backend)
      |> Nx.sqrt()

    assert Nx.to_flat_list(t) == [0.875, 1.75]
  end
end
